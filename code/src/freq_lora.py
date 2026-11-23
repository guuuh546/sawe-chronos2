import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _compute_spectral_entropy(magnitude: torch.Tensor) -> torch.Tensor:
    """计算归一化频谱熵 ∈ [0, 1]"""
    probs = magnitude / (magnitude.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
    max_entropy = torch.log(torch.tensor(magnitude.shape[1], dtype=torch.float32,
                                         device=magnitude.device))
    return entropy / (max_entropy + 1e-8)


class SpectralGatingMechanism(nn.Module):
    """频谱引导门控机制 - 动态控制 LoRA 激活比例"""
    
    def __init__(self, hidden_dim: int = 128, num_freq_bins: int = 64,
                 use_entropy: bool = False):
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.use_entropy = use_entropy
        
        # 分离层以支持熵适配
        self.fc1 = nn.Linear(num_freq_bins, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        # 输出层 bias 初始化为 3.0 → sigmoid(3.0) ≈ 0.95
        # 训练初期 gamma ≈ 0.95，行为接近 vanilla_lora
        nn.init.constant_(self.fc2.bias, 3.0)
    
    def extract_spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        从时间序列提取频谱特征
        Returns:
            spectral_feats: [batch_size, num_freq_bins]
        """
        if x.dim() == 3:
            x = x.mean(dim=-1)
        
        magnitude = torch.abs(torch.fft.rfft(x, dim=1))  # [B, K]
        
        K = magnitude.shape[1]
        if K >= self.num_freq_bins:
            return magnitude[:, :self.num_freq_bins]
        return F.pad(magnitude, (0, self.num_freq_bins - K))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算门控因子 γ ∈ (0, 1]"""
        return self.forward_from_features(self.extract_spectral_features(x))
    
    def forward_from_features(self, spectral_feats: torch.Tensor) -> torch.Tensor:
        """由已提取的频谱特征计算门控因子 γ ∈ (0, 1]（避免重复 FFT）"""
        h = torch.relu(self.fc1(spectral_feats))
        data_logit = self.fc2(h) - self.fc2.bias  # 纯数据驱动部分
        bias = self.fc2.bias  # 固定偏置
        
        if self.use_entropy:
            norm_entropy = _compute_spectral_entropy(spectral_feats)  # [B, 1]
            # 高熵(≈1, 频谱平坦) → 门控削弱 → gamma ≈ sigmoid(bias) ≈ 0.95
            # 低熵(≈0, 频谱稀疏) → 门控充分发挥
            data_logit = data_logit * (1.0 - norm_entropy)
        
        logit = data_logit + bias
        gamma = torch.sigmoid(logit)
        gamma = torch.clamp(gamma, 1e-6, 1.0)
        return gamma


class FrequencyAwareLoRA(nn.Module):
    """
    频率感知的 LoRA 层
    集成了：
    1. 频谱引导门控机制
    2. 频谱稀疏正则化
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        num_freq_bins: int = 64,
        lambda_l1: float = 0.0001,
        use_gating: bool = True,
        use_spectral_reg: bool = True,
        use_entropy: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.lambda_l1 = lambda_l1
        self.use_gating = use_gating
        self.use_spectral_reg = use_spectral_reg
        
        # LoRA 低秩矩阵
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # 频谱门控机制
        self.spectral_gate = SpectralGatingMechanism(
            num_freq_bins=num_freq_bins, use_entropy=use_entropy
        ) if use_gating else None
        
        # 保存当前批次的门控因子和频谱特征用于可视化
        self.current_gamma = None
        self.current_spectral_feats = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA 分支
        lora_out = self.lora_B(self.lora_A(x))
        lora_out = lora_out * self.alpha
        
        # 应用频谱门控
        if self.use_gating and self.spectral_gate is not None:
            spectral_feats = self.spectral_gate.extract_spectral_features(x)
            gamma = self.spectral_gate.forward_from_features(spectral_feats)
            self.current_gamma = gamma
            self.current_spectral_feats = spectral_feats
            gamma = gamma.view(-1, 1, 1)
            lora_out = lora_out * gamma
        
        return lora_out
    
    def compute_spectral_regularization(self) -> torch.Tensor:
        """
        计算频域 L1 正则化项
        Args:
            None
        Returns:
            reg_loss: 标量正则化损失
        """
        if not self.use_spectral_reg:
            return torch.tensor(0.0, device=self.lora_A.weight.device)
        
        # 对 LoRA 权重做频域分析
        A_weight = self.lora_A.weight
        B_weight = self.lora_B.weight
        
        # 计算频域表示的 L1 范数
        A_fft = torch.fft.rfft(A_weight, dim=-1)
        B_fft = torch.fft.rfft(B_weight, dim=-1)
        
        reg_loss = self.lambda_l1 * (
            torch.abs(A_fft).sum() + torch.abs(B_fft).sum()
        )
        
        return reg_loss
