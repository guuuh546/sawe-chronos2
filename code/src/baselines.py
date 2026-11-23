"""
基线方法实现
- LoFT-LLM: 频域 LoRA（对 LoRA 权重做 FFT 滤波）
- FourierFT: 频域直接微调（Gao et al. 2024 忠实实现：稀疏随机谱位置 + 2D IDFT）
- Dense Spectral ΔW: 全频带稠密频谱 ΔW（早期实现，仅作消融，不是 FourierFT）
- Full Fine-tuning: 全参数微调
- Linear Probing: 仅训练输出头
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


# ============================================================
# 1. LoFT-LLM: 频域 LoRA
# ============================================================

class LoFTLoRA(nn.Module):
    """
    LoFT-LLM: 对 LoRA 权重做 FFT，在频域保留高信息量分量，IFFT 恢复
    核心区别：FFT 作用于 LoRA 的 A/B 矩阵，而非输入 hidden state
    """

    def __init__(self, in_features, out_features, rank=4, alpha=1.0,
                 keep_ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.keep_ratio = keep_ratio

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def _freq_filter(self, w: torch.Tensor) -> torch.Tensor:
        """在频域保留 keep_ratio 比例的低频分量，裁掉高频"""
        # w: [d_out, d_in]
        w_fft = torch.fft.rfft(w, dim=-1)  # FFT along input dim
        # 只保留前 keep_ratio 的频率分量
        bins = w_fft.shape[-1]
        keep_bins = max(1, int(bins * self.keep_ratio))
        mask = torch.zeros(bins, device=w.device)
        mask[:keep_bins] = 1.0
        w_fft_filtered = w_fft * mask
        w_filtered = torch.fft.irfft(w_fft_filtered, n=w.shape[-1], dim=-1)
        return w_filtered

    def forward(self, x):
        # 在频域对 LoRA 权重做滤波
        A_filt = self._freq_filter(self.lora_A.weight)  # [rank, in_features]
        B_filt = self._freq_filter(self.lora_B.weight)  # [out_features, rank]

        # 使用滤波后的权重
        lora_out = F.linear(x, A_filt)  # [B, S, rank]
        lora_out = F.linear(lora_out, B_filt)  # [B, S, out_features]
        return lora_out * self.alpha


class LoFTInjectedLinear(nn.Module):
    """带有 LoFT-LLM 注入的线性层"""

    def __init__(self, original_linear, rank=4, alpha=1.0, keep_ratio=0.5):
        super().__init__()
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.lora = LoFTLoRA(
            original_linear.in_features,
            original_linear.out_features,
            rank, alpha, keep_ratio,
        )

    def forward(self, x):
        return self.original_linear(x) + self.lora(x)


def inject_loft_llm(model, rank=4, alpha=1.0, keep_ratio=0.5,
                     target_modules=None):
    """注入 LoFT-LLM 到模型"""
    if target_modules is None:
        target_modules = ["linear"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(t in name.lower() for t in target_modules):
                parent = model
                for part in name.split(".")[:-1]:
                    if part:
                        parent = getattr(parent, part)

                injected = LoFTInjectedLinear(module, rank, alpha, keep_ratio)
                setattr(parent, name.split(".")[-1], injected)

    return model


# ============================================================
# 2. FourierFT: 频域直接微调
#   2a. FourierFTFaithfulLinear —— Gao et al. (ICML 2024) 忠实实现（主实验）
#   2b. DenseSpectralDeltaWLinear —— 早期"全频带稠密频谱 ΔW"实现（仅作消融，
#       不是 FourierFT）
# ============================================================

# ------------------------------------------------------------
# 2a. 忠实实现：与 HuggingFace PEFT 的 fourierft/layer.py 对齐
# ------------------------------------------------------------

class FourierFTFaithfulLinear(nn.Module):
    """Gao et al. (ICML 2024, arXiv:2405.03003) FourierFT 的忠实实现。

    与官方 PEFT 实现（src/peft/tuners/fourierft/layer.py）逐项对齐：

    1. 谱位置 E（随机、冻结、可复现）：每层用 ``random_loc_seed`` 建立
       ``randperm(out*in)[:n_frequency]``，再拆成 (row, col)。
       注意：官方是**逐层**用同一 seed 生成展平索引，而不是按 2D 坐标全局共享；
       同形状的层因此得到完全相同的索引集合，不同形状的层则在各自的展平
       索引空间取样。本实现沿用官方做法（见 work.md）。
    2. 每层独立可训练系数 c ∈ R^n，标准正态初始化（PEFT ``init_weights=False`` 默认）。
    3. ΔW = Re(ifft2(ToDense(c))) * scaling，``torch.fft.ifft2`` 使用默认
       ``norm='backward'``（含 1/(out*in) 归一化）；α（scaling）在取实部之后施加。
    4. 前向 = original(x) + F.linear(x, ΔW)，bias 只在 original 里加一次。
    """

    def __init__(self, original_linear, n_frequency=1000, scaling=150.0,
                 random_loc_seed=777):
        super().__init__()
        if not isinstance(original_linear, nn.Linear):
            raise TypeError("FourierFTFaithfulLinear 只支持 nn.Linear")
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.n_frequency = int(n_frequency)
        self.scaling = float(scaling)
        self.random_loc_seed = int(random_loc_seed)

        n_positions = self.out_features * self.in_features
        if not 0 < self.n_frequency <= n_positions:
            raise ValueError(
                f"n_frequency={self.n_frequency} 必须位于 (0, {n_positions}]")

        # 与 PEFT 完全一致：CPU generator + randperm，按行优先展开成 (row, col)
        flat_idx = torch.randperm(
            n_positions,
            generator=torch.Generator().manual_seed(self.random_loc_seed),
        )[:self.n_frequency]
        self.register_buffer("freq_row", flat_idx // self.in_features,
                             persistent=False)
        self.register_buffer("freq_col", flat_idx % self.in_features,
                             persistent=False)
        # 行优先展平下标（批量 ToDense 时用；与 row/col 等价）
        self.register_buffer("freq_flat", flat_idx.clone(), persistent=False)

        # 每层独立的可训练系数 c（n 个）
        self.spectrum = nn.Parameter(torch.randn(self.n_frequency))

        # 由 FourierFTFaithfulManager 写入的批量 ΔW 缓存；None 表示走逐层 ifft2
        self._delta_w_cache: Optional[torch.Tensor] = None

    def delta_weight(self) -> torch.Tensor:
        """把稀疏谱 ToDense 后做 2D IDFT，返回实部并按 α 缩放。"""
        dense = torch.zeros(self.out_features, self.in_features,
                            device=self.spectrum.device, dtype=torch.float32)
        dense[self.freq_row, self.freq_col] = self.spectrum.float()
        # norm='backward'（PyTorch 默认）：IDFT 自带 1/(out*in) 归一化，与 PEFT 一致
        delta = torch.fft.ifft2(dense).real * self.scaling
        return delta

    def forward(self, x):
        out = self.original_linear(x)
        delta = self._delta_w_cache
        if delta is None:
            delta = self.delta_weight()
        else:
            # 缓存可能来自别的 dtype/device（模型迁移后），不匹配则回退逐层实现
            if delta.device != self.spectrum.device:
                delta = self.delta_weight()
                self._delta_w_cache = delta
        delta_out = F.linear(x.to(delta.dtype), delta)
        return (out + delta_out).to(out.dtype)


class FourierFTFaithfulManager:
    """把各层的 2D IDFT 合并为**按权重形状分组的批量 ifft2**，并缓存 ΔW。

    动机：120 个目标层 × 每层每次 forward 一次 ``ifft2`` + 一次 768² 零张量分配。
    按 (d_out, d_in) 分组后，Chronos-2 只有 3 组（96×768²、12×3072×768、
    12×768×3072），每组一次批量 ``ifft2(..., dim=(-2,-1))`` 即可算完全部层的 ΔW。

    分组内再按 ``chunk_size`` 切块，避免一次性驻留 96×768×768 complex64（≈452MB）
    在 8GB 卡上叠加其他张量导致 OOM。

    缓存失效：
      - **训练（grad enabled）**：默认 ``train_batched=False``，直接把各层
        ``_delta_w_cache`` 置空，让每层走自己的 ifft2（实测这是最快路径，见
        work.md §12 的基准：批量 ifft2 在 4060 上并不比逐层快）。
        若设 ``train_batched=True`` 则每步重算批量 ΔW；不能用跨步缓存，因为缓存
        张量带着 autograd 图，跨多个 backward 复用会报 "backward through the
        graph a second time"。
      - **推理（no_grad）**：按 ``spectrum._version`` 元组判断参数是否被改动，
        未改动则直接复用缓存（成员冻结后整段评测只算一次）。
    """

    def __init__(self, model: Optional[nn.Module] = None, chunk_size: int = 16,
                 train_batched: bool = False):
        self.groups: dict = {}          # (out,in) -> [FourierFTFaithfulLinear]
        self.chunk_size = int(chunk_size)
        self.train_batched = bool(train_batched)
        self._versions: dict = {}       # (out,in) -> tuple[int]
        self.model: Optional[nn.Module] = None
        self.stats = {"refresh_calls": 0, "group_computes": 0, "layers_computed": 0}
        if model is not None:
            self.collect(model)

    # ---------- 注册 ----------
    def register(self, module: "FourierFTFaithfulLinear") -> None:
        key = (module.out_features, module.in_features)
        self.groups.setdefault(key, []).append(module)

    def collect(self, model: nn.Module) -> "FourierFTFaithfulManager":
        self.groups = {}
        for m in model.modules():
            if isinstance(m, FourierFTFaithfulLinear):
                self.register(m)
        return self

    # ---------- 计算 ----------
    def _compute_group(self, mods: List["FourierFTFaithfulLinear"]) -> None:
        o, i = mods[0].out_features, mods[0].in_features
        device = mods[0].spectrum.device
        scaling = mods[0].scaling
        for m in mods[1:]:
            if m.scaling != scaling:
                raise ValueError("同形状组的 scaling 必须一致（注入时统一设置）")

        for start in range(0, len(mods), self.chunk_size):
            chunk = mods[start:start + self.chunk_size]
            g = len(chunk)
            coef = torch.stack([m.spectrum for m in chunk], dim=0)  # [g, n]
            flat = torch.stack([m.freq_flat for m in chunk], dim=0)  # [g, n]
            # ToDense：每层一行，写入稀疏谱位置
            dense = torch.zeros(g, o * i, dtype=coef.dtype, device=device)
            dense = dense.scatter(1, flat, coef)
            spec = dense.view(g, o, i)
            # 一次批量 2D IDFT（norm='backward'，与逐层 ifft2 相同）
            delta = torch.fft.ifft2(spec, dim=(-2, -1)).real * scaling
            for k, m in enumerate(chunk):
                m._delta_w_cache = delta[k]
            self.stats["group_computes"] += 1
            self.stats["layers_computed"] += g
            del coef, flat, dense, spec, delta

    def refresh(self, force: bool = False) -> None:
        """按需刷新所有组的 ΔW 缓存。"""
        self.stats["refresh_calls"] += 1
        grad_enabled = torch.is_grad_enabled()
        for key, mods in self.groups.items():
            if grad_enabled:
                if not self.train_batched:
                    # 训练走逐层 ifft2：实测比批量快（cuFFT 批量无收益），
                    # 且每层各自建图，autograd 安全
                    for m in mods:
                        m._delta_w_cache = None
                    continue
                self._compute_group(mods)
                self._versions[key] = tuple(m.spectrum._version for m in mods)
                continue
            versions = tuple(m.spectrum._version for m in mods)
            cached = all(m._delta_w_cache is not None
                         and m._delta_w_cache.device == m.spectrum.device
                         for m in mods)
            if not force and cached and self._versions.get(key) == versions:
                continue
            self._compute_group(mods)
            self._versions[key] = versions

    # ---------- 挂到模型 ----------
    def attach(self, model: nn.Module) -> nn.Module:
        """把 refresh() 插到模型顶层 forward 之前（保持 `model(x)` 语义不变）。"""
        self.model = model
        original_forward = model.forward
        manager = self

        def forward_with_refresh(*args, **kwargs):
            manager.refresh()
            return original_forward(*args, **kwargs)

        model.forward = forward_with_refresh
        model.fourierft_manager = self
        return model

    def summary(self) -> dict:
        return {
            "groups": {str(k): len(v) for k, v in self.groups.items()},
            "chunk_size": self.chunk_size,
            "stats": dict(self.stats),
        }


def inject_fourier_ft_faithful(model, n_frequency=1000, scaling=150.0,
                               random_loc_seed=777, target_modules=None,
                               verbose=True, batched=True, chunk_size=16,
                               train_batched=False):
    """把忠实 FourierFT 注入模型的目标线性层，并记录参数量。

    Args:
        n_frequency: 每条目层的可训练谱系数个数（默认 1000）
        scaling: α，施加在 2D IDFT 的实部上（默认 150.0）
        random_loc_seed: 谱位置采样种子（默认 777，与 PEFT 一致）
        target_modules: 目标层名子串列表
        batched: True（默认）时挂 `FourierFTFaithfulManager`：推理（no_grad）阶段
            用按权重形状分组的批量 ifft2 + ΔW 缓存；训练阶段默认仍走逐层 ifft2
            （实测最快）。数值等价，见 test_fourierft_batched.py；
            False 时不挂 manager（纯逐层参考路径，对拍/回归用）。
        train_batched: True 时训练阶段也用批量 ifft2（数值等价，但实测更慢）。

    Returns:
        model（同时写入 ``model.fourierft_report`` 字典）
    """
    if target_modules is None:
        target_modules = ["linear"]

    per_layer = {}
    shape_stats = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(t in name.lower() for t in target_modules):
                parent = model
                for part in name.split(".")[:-1]:
                    if part:
                        parent = getattr(parent, part)

                injected = FourierFTFaithfulLinear(
                    module, n_frequency, scaling, random_loc_seed)
                # 与 PEFT 一致：把适配器放到被包裹层的设备上，避免注入后
                # 忘记 model.to(device) 导致 CPU/GPU 混用
                injected = injected.to(device=module.weight.device)
                setattr(parent, name.split(".")[-1], injected)
                per_layer[name] = int(injected.spectrum.numel())
                shape = tuple(module.weight.shape)
                shape_stats[shape] = shape_stats.get(shape, 0) + 1

    adapter_params = int(sum(per_layer.values()))
    # 谱位置索引 E：官方按"同 seed + 同形状 ⇒ 同一索引集"生成，因此按权重重形
    # 去重后实际需要的存储是 2n × 形状数（Chronos-2：3 种形状 → 6000），
    # 而不是 2n × 层数。本实现为每层各存一份缓冲区（总元素 2n×L），
    # 两者都记录，避免论文里把 2n 说成总开销。
    n_shapes = len(shape_stats)

    manager = None
    if batched and per_layer:
        manager = FourierFTFaithfulManager(chunk_size=chunk_size,
                                           train_batched=train_batched)
        manager.collect(model)
        manager.attach(model)

    report = {
        "n_layers": len(per_layer),
        "n_frequency": int(n_frequency),
        "scaling": float(scaling),
        "random_loc_seed": int(random_loc_seed),
        "per_layer_params": per_layer,
        "adapter_params": adapter_params,
        "weight_shapes": {str(k): v for k, v in shape_stats.items()},
        "n_distinct_shapes": n_shapes,
        # 去重后的索引存储（论文口径）
        "frozen_index_entries": 2 * int(n_frequency) * n_shapes,
        # 实现中每层各存一份缓冲区时的实际元素数
        "index_buffer_entries": 2 * int(n_frequency) * len(per_layer),
        "model_trainable_params": int(sum(
            p.numel() for p in model.parameters() if p.requires_grad)),
        "batched_ifft2": bool(manager is not None),
        "ifft2_groups": ({str(k): len(v) for k, v in manager.groups.items()}
                         if manager is not None else {}),
        "ifft2_chunk_size": (manager.chunk_size if manager is not None else None),
    }
    model.fourierft_report = report

    if verbose:
        print(f"[FourierFT] 目标层 {report['n_layers']} 个 × n={n_frequency} "
              f"→ 可训练谱系数 {adapter_params}；冻结索引 E（按形状去重）2n×"
              f"{n_shapes}={report['frozen_index_entries']}，"
              f"逐层缓冲区 {report['index_buffer_entries']}；α={scaling}, "
              f"seed={random_loc_seed}")
        for shape, cnt in sorted(shape_stats.items(), key=lambda z: -z[1]):
            print(f"[FourierFT]   weight shape {shape}: {cnt} 层")
        if manager is not None:
            print(f"[FourierFT] 批量 ifft2：{len(manager.groups)} 组，"
                  f"chunk={manager.chunk_size}，"
                  f"分组={ {tuple(k): len(v) for k, v in manager.groups.items()} }")
        else:
            print("[FourierFT] 逐层 ifft2（参考路径，未挂 manager）")
        print(f"[FourierFT] 注入后模型全部可训练参数: "
              f"{report['model_trainable_params']}")

    return model


# ------------------------------------------------------------
# 2b. 旧实现（全频带稠密频谱 ΔW），仅作附录消融
# ------------------------------------------------------------

class DenseSpectralDeltaWLinear(nn.Module):
    """
    [NOT FourierFT] 全频带稠密频谱 ΔW（本工程早期实现，仅作消融保留）

    训练参数为稠密半谱 (delta_real / delta_imag)，经 1D irfft 重建 ΔW，
    前向做 W0·x + ΔW·x。freq_band='all' 时**没有任何低频截断**，keep_ratio
    实际不生效；每层参数量为 2·d_out·(d_in/2+1)（Chronos-2 的 72 个目标层
    合计约 1.14 亿）。

    这不是 Gao et al. (ICML 2024) 的 FourierFT（后者是"共享随机稀疏谱位置 +
    每层 n 个可训练系数 + 2D IDFT + α 缩放"）。需要该基线请使用
    FourierFTFaithfulLinear / inject_fourier_ft_faithful。
    """

    def __init__(self, original_linear, keep_ratio=0.5, freq_band='all'):
        super().__init__()
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False

        out_f, in_f = original_linear.weight.shape
        full_bins = in_f // 2 + 1
        self.full_bins = full_bins
        self.freq_band = freq_band

        # 根据波段确定保留的频率范围
        total_keep = max(1, int(full_bins * keep_ratio))
        if freq_band == 'low':
            self.start_bin = 0
            self.end_bin = total_keep
        elif freq_band == 'high':
            self.start_bin = full_bins - total_keep
            self.end_bin = full_bins
        elif freq_band == 'mid':
            mid_start = (full_bins - total_keep) // 2
            self.start_bin = mid_start
            self.end_bin = mid_start + total_keep
        else:  # all
            self.start_bin = 0
            self.end_bin = full_bins

        self.fft_bins = self.end_bin - self.start_bin
        # 实数 + 虚数部分作为可训练参数
        self.delta_real = nn.Parameter(
            torch.zeros(out_f, self.fft_bins) * 0.01)
        self.delta_imag = nn.Parameter(
            torch.zeros(out_f, self.fft_bins) * 0.01)

    def forward(self, x):
        # 构造完整频域表示
        full_bins = self.full_bins
        delta_fft = torch.complex(
            torch.zeros(self.delta_real.shape[0], full_bins,
                        device=x.device),
            torch.zeros(self.delta_real.shape[0], full_bins,
                        device=x.device),
        )
        delta_fft[:, self.start_bin:self.end_bin] = torch.complex(
            self.delta_real, self.delta_imag)

        # IFFT 得到时域 ΔW
        delta_w = torch.fft.irfft(
            delta_fft, n=self.original_linear.weight.shape[-1], dim=-1)

        # 计算输出 = original(x) + ΔW @ x
        # 注：旧版这里把 bias 又传给了 F.linear，导致 bias 被重复计入；
        # Chronos 目标层 bias=False，故历史数值不受影响，此处修正。
        original_out = self.original_linear(x)
        delta_out = F.linear(x, delta_w)
        return original_out + delta_out


def inject_dense_spectral_dw(model, keep_ratio=0.5, target_modules=None,
                             freq_band='all'):
    """注入"全频带稠密频谱 ΔW"（旧实现，不是 Gao et al. FourierFT）

    Args:
        freq_band: 'low' | 'mid' | 'high' | 'all'
            选择训练哪个频率波段的分量（默认 'all'，即不截断）
    """
    if target_modules is None:
        target_modules = ["linear"]

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(t in name.lower() for t in target_modules):
                parent = model
                for part in name.split(".")[:-1]:
                    if part:
                        parent = getattr(parent, part)

                injected = DenseSpectralDeltaWLinear(
                    module, keep_ratio, freq_band)
                setattr(parent, name.split(".")[-1], injected)

    return model


# 兼容别名（DEPRECATED）：仅为不破坏旧脚本而保留，调用即发出 DeprecationWarning。
# 它们指向的是"全频带稠密频谱 ΔW"消融实现，**不是** Gao et al. FourierFT；
# 主实验/论文复算请用 inject_fourier_ft_faithful。
def FourierFTLinear(*args, **kwargs):
    warnings.warn(
        "FourierFTLinear 已废弃：它是全频带稠密频谱 ΔW 消融实现，不是 Gao et al. "
        "FourierFT。请改用 FourierFTFaithfulLinear。",
        DeprecationWarning, stacklevel=2)
    return DenseSpectralDeltaWLinear(*args, **kwargs)


def inject_fourier_ft(*args, **kwargs):
    warnings.warn(
        "inject_fourier_ft 已废弃：它注入的是稠密频谱 ΔW 消融实现，不是 Gao et al. "
        "FourierFT。请改用 inject_fourier_ft_faithful。",
        DeprecationWarning, stacklevel=2)
    return inject_dense_spectral_dw(*args, **kwargs)


# ============================================================
# 3. Full Fine-tuning: 全参数微调
# ============================================================

def setup_full_ft(model: nn.Module) -> nn.Module:
    """解冻所有参数，允许全参数微调"""
    for param in model.parameters():
        param.requires_grad = True
    return model


# ============================================================
# 4. Linear Probing: 仅训练输出头
# ============================================================

def setup_linear_probe(model: nn.Module,
                       head_names=("output_head", "pred_head")) -> nn.Module:
    """
    冻结所有参数，仅解冻输出头
    head_names: 输出头子模块的名称
    """
    # 先全部冻结
    for param in model.parameters():
        param.requires_grad = False

    # 解冻输出头
    for name, param in model.named_parameters():
        if any(h in name for h in head_names):
            param.requires_grad = True

    return model
