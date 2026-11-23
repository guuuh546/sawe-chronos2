import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导出基础模型
from src.foundation_models import Chronos2ForecastingModel, TimesFMForecastingModel  # noqa: F401


class SimpleTimeSeriesModel(nn.Module):
    """
    简单的时序模型（避免使用 Transformer 的复杂内部结构）
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 32,
        seq_len: int = 48,
        pred_len: int = 24,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 简单的前馈网络
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layer1 = nn.Linear(d_model, d_model * 2)
        self.layer2 = nn.Linear(d_model * 2, d_model)
        self.output_proj = nn.Linear(d_model, input_dim)
        
        # 预测长度变换
        self.pred_head = nn.Linear(seq_len, pred_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            preds: [batch_size, pred_len, input_dim]
        """
        # 输入投影
        x_emb = self.input_proj(x)
        
        # 前馈层
        x = F.relu(self.layer1(x_emb))
        x = self.layer2(x) + x_emb  # 残差连接
        
        # 输出投影
        x_out = self.output_proj(x)
        
        # 预测长度变换
        x_out = x_out.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x_out = self.pred_head(x_out)  # [batch, input_dim, pred_len]
        preds = x_out.permute(0, 2, 1)  # [batch, pred_len, input_dim]
        
        return preds


def get_target_modules(backbone: str):
    """返回指定基础模型需要注入 LoRA 的目标层名模式。

    Args:
        backbone: "chronos2" / "timesfm" / "simple"
    """
    b = backbone.lower()
    if b in ("chronos", "chronos-2", "chronos2"):
        return ["self_attention", "mlp"]
    if b == "timesfm":
        return ["qkv_proj", "o_proj", "gate_proj", "down_proj"]
    # 旧的自建小 MLP 验证模型
    return ["proj", "layer", "head"]


def build_backbone(backbone: str = "chronos2", input_dim: int = 1,
                   seq_len: int = 96, pred_len: int = 96,
                   model_path: str = "loratime"):
    """构建开源时序基础模型。

    Args:
        backbone: "chronos2"（默认，Amazon Chronos-2）或 "timesfm"（Google TimesFM）。
        model_path: 本地权重根目录（包含 chronos-2 / timesfm 子目录）。
    """
    b = backbone.lower()
    if b in ("chronos", "chronos-2", "chronos2"):
        return Chronos2ForecastingModel(
            model_path=os.path.join(model_path, "chronos-2"),
            input_dim=input_dim, seq_len=seq_len, pred_len=pred_len)
    if b == "timesfm":
        return TimesFMForecastingModel(
            model_path=os.path.join(model_path, "timesfm"),
            input_dim=input_dim, seq_len=seq_len, pred_len=pred_len)
    if b in ("simple", "mlp"):
        return SimpleTimeSeriesModel(
            input_dim=input_dim, d_model=32, seq_len=seq_len, pred_len=pred_len)
    raise ValueError(f"未知基础模型: {backbone}")
