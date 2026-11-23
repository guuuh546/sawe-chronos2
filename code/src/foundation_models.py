"""
基础模型接入模块
支持加载 Chronos-2 和 TimesFM 预训练时序模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch as st
import json
import os
from typing import Optional


class ChronosSelfAttention(nn.Module):
    """Chronos-2 自注意力层（无 bias 的 Q/K/V/O 投影）"""

    def __init__(self, d_model: int = 768, num_heads: int = 12):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, S, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, S, self.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        Q = self._split_heads(self.q(hidden_states))
        K = self._split_heads(self.k(hidden_states))
        V = self._split_heads(self.v(hidden_states))

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = self._merge_heads(attn_output)
        return self.o(attn_output)


class ChronosMLP(nn.Module):
    """Chronos-2 前馈网络层（无 bias）"""

    def __init__(self, d_model: int = 768, d_ff: int = 3072):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi(x)))


class ChronosLayerNorm(nn.Module):
    """简化的 LayerNorm（兼容 Chronos-2 权重）"""

    def __init__(self, d_model: int = 768, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class ChronosBlock(nn.Module):
    """Chronos-2 Encoder Block: self_attn_0 -> self_attn_1 -> mlp"""

    def __init__(self, d_model: int = 768, num_heads: int = 12, d_ff: int = 3072):
        super().__init__()
        self.layer = nn.ModuleList([
            ChronosSelfAttentionLayer(d_model, num_heads),
            ChronosSelfAttentionLayer(d_model, num_heads),
            ChronosFFNLayer(d_model, d_ff),
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for sublayer in self.layer:
            hidden_states = sublayer(hidden_states)
        return hidden_states


class ChronosSelfAttentionLayer(nn.Module):
    """带 LayerNorm + 残差的自注意力子层"""

    def __init__(self, d_model: int = 768, num_heads: int = 12):
        super().__init__()
        self.layer_norm = ChronosLayerNorm(d_model)
        self.self_attention = ChronosSelfAttention(d_model, num_heads)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.self_attention(hidden_states)
        return residual + hidden_states


class ChronosFFNLayer(nn.Module):
    """带 LayerNorm + 残差的 FFN 子层"""

    def __init__(self, d_model: int = 768, d_ff: int = 3072):
        super().__init__()
        self.layer_norm = ChronosLayerNorm(d_model)
        self.mlp = ChronosMLP(d_model, d_ff)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Chronos2Encoder(nn.Module):
    """Chronos-2 Encoder（12 个 block）"""

    def __init__(self, num_layers: int = 12, d_model: int = 768,
                 num_heads: int = 12, d_ff: int = 3072):
        super().__init__()
        self.block = nn.ModuleList([
            ChronosBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            hidden_states = block(hidden_states)
        return hidden_states


class Chronos2ForecastingModel(nn.Module):
    """
    基于 Chronos-2 的时序预测模型
    输入:  [B, seq_len, input_dim] -> 输出:  [B, pred_len, input_dim]
    """

    def __init__(self, model_path="loratime/chronos-2", input_dim=1,
                 seq_len=96, pred_len=96, d_model=768):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim

        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        num_layers = config.get("num_layers", 12)
        num_heads = config.get("num_heads", 12)
        d_ff = config.get("d_ff", 3072)

        self.encoder = Chronos2Encoder(num_layers, d_model, num_heads, d_ff)
        self.input_proj = nn.Linear(input_dim, d_model)
        # 投影后归一化，稳定 7→768 投影的输出分布，使其更接近冻结 encoder 的 pre-norm 期望
        self.input_norm = nn.LayerNorm(d_model)
        # 可学习的位置编码：原 Chronos-2 用 RoPE，这里用绝对位置编码补齐时序顺序信息
        # （encoder 本身不含位置编码，缺了它序列顺序信息会丢失）
        self.position_embedding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, pred_len * input_dim),
        )

        self._load_pretrained_weights(model_path)
        self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, model_path: str):
        weights_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(weights_path):
            print(f"[WARNING] 权重文件不存在: {weights_path}")
            return
        try:
            state_dict = {}
            with st.safe_open(weights_path, framework="pt") as f:
                for key in f.keys():
                    if key.startswith("encoder."):
                        state_dict[key] = f.get_tensor(key)
            missing, _ = self.load_state_dict(state_dict, strict=False)
            matched = len(state_dict) - len(missing)
            print(f"[INFO] Chronos-2 权重加载: {matched}/{len(state_dict)} keys")
        except Exception as e:
            print(f"[WARNING] Chronos-2 权重加载失败: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.input_norm(self.input_proj(x))
        x = x + self.position_embedding[:, :x.shape[1], :]
        x = self.encoder(x)
        x = x[:, -1, :]  # 取最后一个 token，保留时序位置信息
        x = self.output_head(x)
        return x.view(B, self.pred_len, self.input_dim)


# ============================================================
# TimesFM 基础模型 (50层 Transformer, 1280 hidden, 16 heads)
# ============================================================

class TimeFMRMSNorm(nn.Module):
    """TimesFM RMSNorm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class TimeFMAttention(nn.Module):
    """TimesFM Multi-Head Attention with fused QKV projection"""

    def __init__(self, d_model=1280, num_heads=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, _, S, _ = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, S, self.d_model)

    def forward(self, hidden_states):
        B, S, _ = hidden_states.shape
        Q, K, V = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        Q, K, V = self._split_heads(Q), self._split_heads(K), self._split_heads(V)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(S, S, device=hidden_states.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return self.o_proj(self._merge_heads(torch.matmul(attn, V)))


class TimeFMMLP(nn.Module):
    """TimesFM FFN (gate_proj + down_proj)"""

    def __init__(self, d_model=1280):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.down_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * x)


class TimeFMBlock(nn.Module):
    """TimesFM Transformer Block"""

    def __init__(self, d_model=1280, num_heads=16):
        super().__init__()
        self.input_layernorm = TimeFMRMSNorm(d_model)
        self.self_attn = TimeFMAttention(d_model, num_heads)
        self.post_attn_ln = TimeFMRMSNorm(d_model)
        self.mlp = TimeFMMLP(d_model)
        self.mlp_layer_norm = TimeFMRMSNorm(d_model)

    def forward(self, hidden_states):
        # Pre-norm Attention + Residual
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = residual + self.post_attn_ln(hidden_states)
        # Pre-norm FFN + Residual
        residual = hidden_states
        hidden_states = self.mlp(self.mlp_layer_norm(hidden_states))
        return residual + hidden_states


class TimesFMForecastingModel(nn.Module):
    """
    基于 TimesFM checkpoint 的时序预测模型
    50层 Transformer, 1280 hidden dim, 16 heads

    输入:  [B, seq_len, input_dim]
    输出:  [B, pred_len, input_dim]
    """

    def __init__(self, model_path="loratime/timesfm", input_dim=1,
                 seq_len=96, pred_len=96, d_model=1280,
                 num_layers=50, num_heads=16):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.layers = nn.ModuleList([
            TimeFMBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.final_layernorm = TimeFMRMSNorm(d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, pred_len * input_dim),
        )

        self._load_pretrained_weights(model_path)
        self.freeze_transformer()

    def freeze_transformer(self):
        for param in self.layers.parameters():
            param.requires_grad = False
        for param in self.final_layernorm.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, model_path: str):
        ckpt_path = os.path.join(model_path, "torch_model.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] TimesFM 权重文件不存在: {ckpt_path}")
            return
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            remapped = {}
            for ckpt_key, tensor in ckpt.items():
                new_key = self._remap_key(ckpt_key)
                if new_key:
                    remapped[new_key] = tensor

            missing, unexpected = self.load_state_dict(remapped, strict=False)
            matched = len(remapped) - len(missing)
            print(f"[INFO] TimesFM 权重加载: {matched}/{len(remapped)} keys"
                  + (f" (missing: {len(missing)})" if missing else ""))
        except Exception as e:
            print(f"[WARNING] TimesFM 权重加载失败: {e}")

    def _remap_key(self, ckpt_key: str):
        if ckpt_key.startswith("stacked_transformer.layers."):
            parts = ckpt_key.split(".", 3)
            if len(parts) >= 4:
                layer_idx = parts[2]
                rest = parts[3]
                if "self_attn.scaling" in rest:
                    return None
                rest = rest.replace("mlp.layer_norm", "mlp_layer_norm")
                return f"layers.{layer_idx}.{rest}"

        if ckpt_key.startswith("freq_emb."):
            return None

        if ckpt_key.startswith("input_ff_layer."):
            return None  # Skip: internal patching weights, different dims

        if ckpt_key.startswith("horizon_ff_layer."):
            return None  # Skip: internal patching weights

        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layernorm(x)
        x = x.mean(dim=1)
        x = self.output_head(x)
        return x.view(B, self.pred_len, self.input_dim)
