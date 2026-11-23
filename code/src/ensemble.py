"""
频谱感知动态集成模型
================================
核心组件：
1. SpectrumRouter —— 频谱感知路由器，根据输入频谱特征预测各基模型权重
2. SpectrumAwareEnsemble —— 完整的动态集成框架，支持 Early-Exit 推理
3. 软标签训练策略 —— 让路由器学习"哪个模型在当前样本上更好"，而不是直接拟合 MSE

关键设计决策（回应审稿人）：
- 不是"数据集级"选择，而是"样本级（instance-level）"选择
- 路由器训练用软标签（每个样本的最优模型相对误差），避免隧道视野
- 支持 Early-Exit，推理时若某模型权重 > 阈值则跳过其余模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from src.spectral_features import MultiVariateSpectralExtractor


class SpectrumRouter(nn.Module):
    """
    频谱感知路由器
    输入：多变量时间序列 [B, S, D]
    输出：各基模型的权重 [B, M]（M = 基模型数量）
    """

    def __init__(self, n_features: int, seq_len: int = 96,
                 num_models: int = 3, hidden_dim: int = 128,
                 use_csd: bool = True, use_stats: bool = True,
                 temperature: float = 1.0):
        super().__init__()
        self.num_models = num_models
        self.temperature = temperature

        self.spectral_extractor = MultiVariateSpectralExtractor(
            n_features=n_features,
            seq_len=seq_len,
            output_dim=hidden_dim,
            use_csd=use_csd,
            use_stats=use_stats,
        )

        self.router_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_models),
        )

        nn.init.zeros_(self.router_head[-1].weight)
        nn.init.zeros_(self.router_head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
        Returns:
            weights: [B, M]  归一化权重（Softmax）
        """
        spectral_feat = self.spectral_extractor(x)  # [B, hidden_dim]
        logits = self.router_head(spectral_feat)  # [B, M]
        weights = F.softmax(logits / self.temperature, dim=-1)  # [B, M]
        return weights

    def get_soft_targets(self, preds_list: List[torch.Tensor],
                         target: torch.Tensor) -> torch.Tensor:
        """
        计算软标签：每个模型在当前样本上的"相对优劣"
        用负误差（MSE 的倒数）作为权重，误差越小权重越大
        然后做 Softmax 归一化，得到"理想的集成权重"

        Args:
            preds_list: 每个模型的预测 [B, S, D]，共 M 个
            target: 真实值 [B, S, D]
        Returns:
            soft_targets: [B, M]  软标签（理想权重分布）
        """
        M = len(preds_list)
        errors = []
        for pred in preds_list:
            mse_per_sample = ((pred - target) ** 2).mean(dim=(1, 2))  # [B]
            errors.append(mse_per_sample)

        errors = torch.stack(errors, dim=-1)  # [B, M]

        error_min = errors.min(dim=-1, keepdim=True).values
        error_rel = errors / (error_min + 1e-8)  # 相对最优模型的倍数

        scores = -error_rel  # 负相对误差，越小越好 → 分数越高
        soft_targets = F.softmax(scores / self.temperature, dim=-1)  # [B, M]

        return soft_targets.detach()


class SpectrumAwareEnsemble(nn.Module):
    """
    频谱感知动态集成模型
    支持：
    - 加权平均集成
    - Early-Exit 推理（某模型权重 > 阈值时跳过其余）
    - 硬选择（取权重最大的模型）
    """

    def __init__(self, models: List[nn.Module],
                 router: SpectrumRouter,
                 early_exit_threshold: float = 0.9,
                 mode: str = "weighted"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.router = router
        self.early_exit_threshold = early_exit_threshold
        self.mode = mode  # "weighted" | "hard"
        self.num_models = len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准前向传播（加权平均）
        Args:
            x: [B, S, D]
        Returns:
            output: [B, S, D]
        """
        weights = self.router(x)  # [B, M]

        preds = []
        for model in self.models:
            preds.append(model(x))
        preds = torch.stack(preds, dim=1)  # [B, M, S, D]

        if self.mode == "hard":
            _, max_idx = weights.max(dim=-1)  # [B]
            batch_idx = torch.arange(x.shape[0], device=x.device)
            output = preds[batch_idx, max_idx]  # [B, S, D]
        else:
            weights_exp = weights.unsqueeze(-1).unsqueeze(-1)  # [B, M, 1, 1]
            output = (preds * weights_exp).sum(dim=1)  # [B, S, D]

        return output

    @torch.no_grad()
    def forward_with_early_exit(self, x: torch.Tensor
                                ) -> Tuple[torch.Tensor, Dict]:
        """
        带 Early-Exit 的推理（用于实际部署和论文效率分析）
        如果某个模型的权重 > threshold，只跑那一个模型

        Returns:
            output: [B, S, D]
            info: dict 包含 exit 统计信息
        """
        weights = self.router(x)  # [B, M]
        B = x.shape[0]

        max_w, max_idx = weights.max(dim=-1)  # [B]
        exit_mask = max_w >= self.early_exit_threshold  # [B]  哪些样本触发早退

        output = torch.zeros_like(self.models[0](x[:1])).repeat(B, 1, 1)  # [B, S, D]
        models_used = torch.ones(B, dtype=torch.long, device=x.device) * self.num_models

        # 处理触发早退的样本
        if exit_mask.any():
            exit_samples = x[exit_mask]
            exit_model_indices = max_idx[exit_mask]
            unique_models = exit_model_indices.unique()

            for m_idx in unique_models:
                m_mask = exit_model_indices == m_idx
                m_samples = exit_samples[m_mask]
                m_pred = self.models[m_idx.item()](m_samples)

                full_mask = exit_mask.clone()
                full_mask[exit_mask] = m_mask
                output[full_mask] = m_pred

            models_used[exit_mask] = 1

        # 处理未触发早退的样本
        if (~exit_mask).any():
            noexit_samples = x[~exit_mask]
            noexit_preds = []
            for model in self.models:
                noexit_preds.append(model(noexit_samples))
            noexit_preds = torch.stack(noexit_preds, dim=1)  # [B2, M, S, D]
            noexit_weights = weights[~exit_mask].unsqueeze(-1).unsqueeze(-1)
            noexit_output = (noexit_preds * noexit_weights).sum(dim=1)
            output[~exit_mask] = noexit_output

        info = {
            "exit_ratio": exit_mask.float().mean().item(),
            "avg_models_used": models_used.float().mean().item(),
            "weights": weights.detach().cpu().numpy(),
            "max_weight": max_w.detach().cpu().numpy(),
        }

        return output, info

    def router_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算路由器的训练损失（KL 散度，拟合软标签）
        Args:
            x: [B, S, D]
            y: [B, S, D]
        Returns:
            loss: 标量
        """
        with torch.no_grad():
            preds_list = [m(x) for m in self.models]
            soft_targets = self.router.get_soft_targets(preds_list, y)

        pred_weights = self.router(x)  # [B, M]

        # KL 散度：让预测权重逼近软标签
        kl = F.kl_div(
            torch.log(pred_weights + 1e-8),
            soft_targets,
            reduction="batchmean"
        )
        return kl


def train_router(ensemble: SpectrumAwareEnsemble,
                 train_loader, val_loader,
                 lr: float = 1e-3, num_epochs: int = 20,
                 device: str = "cpu",
                 early_stopping_patience: int = 5,
                 use_val_for_training: bool = True) -> Dict:
    """
    训练路由器（冻结所有基模型，只训路由器）
    使用 KL 散度 + 软标签策略，避免隧道视野

    Args:
        use_val_for_training: 是否用验证集训练路由器。
            由于基模型在训练集上过拟合程度不同，训练集上的模型排名
            可能与测试集相反，用验证集训练路由器泛化更好。
    """
    for model in ensemble.models:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    ensemble.router.train()
    optimizer = torch.optim.AdamW(ensemble.router.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    if use_val_for_training:
        router_train_loader = val_loader
        router_val_loader = train_loader
    else:
        router_train_loader = train_loader
        router_val_loader = val_loader

    for epoch in range(num_epochs):
        train_loss_sum = 0.0
        train_batches = 0

        ensemble.router.train()
        for batch in router_train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = ensemble.router_loss(x, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # 验证
        val_loss_sum = 0.0
        val_batches = 0
        ensemble.router.eval()
        with torch.no_grad():
            for batch in router_val_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                loss = ensemble.router_loss(x, y)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train KL: {avg_train_loss:.4f}, Val KL: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history


# ============================================================
# 路由器训练的缓存加速（数值等价于 train_router）
# ============================================================
# 原实现在每个 epoch 的每一步都对 3 个**冻结**成员重新前向一次（`router_loss`），
# 这部分计算在 20 个 epoch × 上千 step 下完全重复。缓存版把成员的 (x, y, preds)
# 按数据集原顺序算一遍存起来，训练时只重复 `router(x)` 与 KL，从而把路由器训练
# 从"O(epochs × batches × 3 次成员前向)"降到"O(epochs × batches × 1 次 router 前向)"。
#
# 等价性要点（务必保持）：
#   1. 成员在缓存构建时置 eval() 且 no_grad —— 与原 `train_router` 里
#      `model.eval()` + `with torch.no_grad()` 的成员前向完全一致；
#   2. 缓存按数据集索引顺序构建（shuffle=False 的迭代不消耗随机数），
#      因此缓存构建不改变全局 RNG 状态；训练时用与原始 loader 相同的
#      batch_size / shuffle 重建 DataLoader，随机置乱序列与原来逐步相同；
#   3. 优化器（AdamW, lr）、epoch 数、早停耐心、τ、KL(reduction='batchmean')
#      与损失次序完全照搬。


class _RouterCacheDataset(Dataset):
    """把缓存的 (x, y, preds, raw_feat) 按样本暴露给 DataLoader（置乱顺序与原 loader 一致）。"""

    def __init__(self, X, Y: torch.Tensor, P: torch.Tensor,
                 R: Optional[torch.Tensor] = None):
        self.X, self.Y, self.P, self.R = X, Y, P, R  # X:[N,S,D] Y:[N,T,D] P:[M,N,T,D] R:[N,F]

    def __len__(self) -> int:
        return self.Y.shape[0]

    def __getitem__(self, i: int):
        # DataLoader 会沿新维堆叠 → 训练循环里再转置回 [M, B, T, D]
        x = None if self.X is None else self.X[i]
        r = None if self.R is None else self.R[i]
        return x, self.Y[i], self.P[:, i], r


def _router_cache_collate(batch):
    """与 default_collate 等价，但允许 X / raw_feat 为 None（drop_x 后 X 不存在）。"""
    x0 = batch[0][0]
    xs = None if x0 is None else torch.stack([b[0] for b in batch], 0)
    ys = torch.stack([b[1] for b in batch], 0)
    ps = torch.stack([b[2] for b in batch], 0)
    r0 = batch[0][3]
    rs = None if r0 is None else torch.stack([b[3] for b in batch], 0)
    return xs, ys, ps, rs


class RouterCache:
    """成员预测缓存。CPU float32 存储，逐 batch 搬回设备后数值与原前向逐位一致。

    ``sampler`` 保存原 DataLoader 的采样器对象：缓存版训练复用**同一个**采样器，
    以便每次 ``iter()`` 抽取到的置乱序列与原实现完全相同（RandomSampler 每次
    迭代都从全局 RNG 重新抽 randperm，因此只要 RNG 状态与调用次数一致，顺序就一致）。
    """

    def __init__(self, X, Y: torch.Tensor, P: torch.Tensor,
                 batch_size: int, sampler=None,
                 raw_feat: Optional[torch.Tensor] = None):
        self.X, self.Y, self.P = X, Y, P
        self.raw_feat = raw_feat
        self.batch_size = int(batch_size)
        self.sampler = sampler

    @property
    def n_samples(self) -> int:
        return int(self.Y.shape[0])

    @property
    def n_members(self) -> int:
        return int(self.P.shape[0])

    @property
    def shuffle(self) -> bool:
        return isinstance(self.sampler, torch.utils.data.RandomSampler)

    def subset(self, indices) -> "RouterCache":
        """取成员子集（用于成员组合消融），共享 X/Y/raw_feat 内存。"""
        return RouterCache(self.X, self.Y, self.P[list(indices)],
                           self.batch_size, self.sampler, self.raw_feat)

    def with_raw(self, extractor, device: str = "cpu", chunk: int = 256) -> "RouterCache":
        """按当前 cache 的样本顺序计算 raw 频谱特征（共享 X/Y/P 内存）。

        `use_csd/use_stats` 不同的 extractor 输出维度不同，必须各自调用一次。
        用较小 chunk 控制峰值显存（eigvalsh 的中间张量是 [B, D, D]）。
        """
        if self.X is None:
            raise RuntimeError("X 已被释放，无法计算 raw_feat")
        chunks = []
        with torch.no_grad():
            for start in range(0, self.n_samples, chunk):
                xb = self.X[start:start + chunk].to(device)
                chunks.append(extractor.extract_raw(xb).cpu())
        return RouterCache(self.X, self.Y, self.P, self.batch_size,
                           self.sampler, torch.cat(chunks, 0))

    def drop_x(self) -> None:
        """释放原始输入（路由器训练只用 raw_feat；X 往往是最大的那块）。"""
        self.X = None

    def memory_bytes(self) -> dict:
        def nbytes(t):
            return 0 if t is None else int(t.numel() * t.element_size())
        return {"X": nbytes(self.X), "Y": nbytes(self.Y), "P": nbytes(self.P),
                "raw_feat": nbytes(self.raw_feat),
                "total": nbytes(self.X) + nbytes(self.Y) + nbytes(self.P)
                + nbytes(self.raw_feat)}

    def loader(self) -> DataLoader:
        # 复用原采样器对象 ⇒ 置乱序列与原始 loader 每次 iter() 完全一致
        return DataLoader(_RouterCacheDataset(self.X, self.Y, self.P, self.raw_feat),
                          batch_size=self.batch_size, sampler=self.sampler,
                          collate_fn=_router_cache_collate)


@torch.no_grad()
def build_router_cache(models: List[nn.Module], loader, device: str = "cpu",
                       router: Optional[SpectrumRouter] = None,
                       keep_x: bool = True) -> RouterCache:
    """按数据集索引顺序缓存成员在 loader 上的预测（可选：同时缓存 raw 频谱特征）。

    Args:
        models: 成员模型（会被置为 eval()，不改变参数）
        loader: 原始 DataLoader；取其 dataset / batch_size / sampler 配置
        device: 构建缓存时前向所用设备
        router: 若给出，则用其 `spectral_extractor.extract_raw(x)` 一并缓存 raw_feat，
            避免 router 每个 epoch 重复算 PSD/eigvalsh（ETTm2 上这是主要瓶颈）
        keep_x: 是否保留原始输入 X（router 训练不需要；置 False 可省大量内存）

    Returns:
        RouterCache（`raw_feat` 为 None 表示未缓存特征，old 路径仍可用）
    """
    for m in models:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    batch_size = loader.batch_size or 32
    extractor = router.spectral_extractor if router is not None else None

    # 每次迭代 DataLoader 都会从全局 RNG 抽一个 base_seed（即使 num_workers=0），
    # 若不还原，会让后续 RandomSampler 的置乱序列偏移。这里把 RNG 状态做成
    # "进入-离开完全一致"，使缓存构建对训练随机性零影响。
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # 关键：按索引顺序（shuffle=False）遍历，不消耗全局 RNG
    ordered = DataLoader(loader.dataset, batch_size=batch_size, shuffle=False)
    Xs, Ys, Ps, Rs = [], [], [], []
    try:
        with torch.no_grad():
            for x, y in ordered:
                x = x.to(device)
                y = y.to(device)
                preds = torch.stack([m(x).detach() for m in models], dim=0)
                if keep_x:
                    Xs.append(x.cpu())
                Ys.append(y.cpu())
                Ps.append(preds.cpu())
                if extractor is not None:
                    Rs.append(extractor.extract_raw(x).cpu())
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    X = torch.cat(Xs, 0) if keep_x else None
    Y = torch.cat(Ys, 0)
    P = torch.cat(Ps, 1)  # [M, N, T, D]
    R = torch.cat(Rs, 0) if Rs else None
    return RouterCache(X, Y, P, batch_size, loader.sampler, R)


def router_loss_cached(ensemble: SpectrumAwareEnsemble,
                      y: torch.Tensor, preds: torch.Tensor,
                      raw_feat: torch.Tensor) -> torch.Tensor:
    """与 `SpectrumAwareEnsemble.router_loss` 相同的 KL 损失，但完全走缓存：

    - 软标签来自缓存成员预测 `preds`（[M, B, T, D]）与 `y`；
    - 路由器权重 = softmax(router_head(compressor(raw_feat)) / τ)，
      与 `router(x)` 逐位相同（`MultiVariateSpectralExtractor.forward` 即
      `compressor(extract_raw(x))`），但省掉每步的 PSD + K 次 eigvalsh。
    """
    preds_list = [preds[i] for i in range(preds.shape[0])]
    soft_targets = ensemble.router.get_soft_targets(preds_list, y)

    extractor = ensemble.router.spectral_extractor
    compressed = extractor.compressor(raw_feat)
    logits = ensemble.router.router_head(compressed)
    pred_weights = F.softmax(logits / ensemble.router.temperature, dim=-1)
    return F.kl_div(
        torch.log(pred_weights + 1e-8),
        soft_targets,
        reduction="batchmean",
    )


def train_router_cached(ensemble: SpectrumAwareEnsemble,
                        train_cache: RouterCache, val_cache: RouterCache,
                        lr: float = 1e-3, num_epochs: int = 20,
                        device: str = "cpu",
                        early_stopping_patience: int = 5,
                        use_val_for_training: bool = True) -> Dict:
    """`train_router` 的缓存版：只消除重复计算（成员前向 + 频谱特征），其余逐项照搬。

    Args:
        train_cache: 对应**训练集**（dls["train"]）的缓存（需含 raw_feat）
        val_cache:   对应**验证集**（dls["val"]）的缓存（需含 raw_feat）
        use_val_for_training: 与 `train_router` 同义——为 True 时用验证集训练
            路由器、用训练集做早停验证（交换两个 cache 的角色）。
    """
    if train_cache.raw_feat is None or val_cache.raw_feat is None:
        raise ValueError("train_router_cached 需要带 raw_feat 的 cache；"
                         "请用 build_router_cache(..., router=router) 构建")

    for model in ensemble.models:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    ensemble.router.train()
    optimizer = torch.optim.AdamW(ensemble.router.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    if use_val_for_training:
        router_train_loader = val_cache.loader()
        router_val_loader = train_cache.loader()
    else:
        router_train_loader = train_cache.loader()
        router_val_loader = val_cache.loader()

    for epoch in range(num_epochs):
        train_loss_sum = 0.0
        train_batches = 0

        ensemble.router.train()
        for _, y, preds, raw in router_train_loader:
            y = y.to(device)
            # contiguous()：恢复与 torch.stack 相同的连续布局，否则 reduce 的
            # 浮点累加顺序会变，导致 KL 出现 ~1e-7 的差异（就不是逐位等价了）
            preds = preds.transpose(0, 1).contiguous().to(device)
            raw = raw.to(device)

            optimizer.zero_grad()
            loss = router_loss_cached(ensemble, y, preds, raw)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        history["train_loss"].append(avg_train_loss)

        val_loss_sum = 0.0
        val_batches = 0
        ensemble.router.eval()
        with torch.no_grad():
            for _, y, preds, raw in router_val_loader:
                y = y.to(device)
                preds = preds.transpose(0, 1).contiguous().to(device)
                raw = raw.to(device)
                loss = router_loss_cached(ensemble, y, preds, raw)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train KL: {avg_train_loss:.4f}, Val KL: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history
