"""
多变量频谱特征提取模块
================================
从多变量时间序列中提取丰富的频域特征，用于频谱感知路由器。

三种特征提取策略（可组合使用）：
1. 逐变量功率谱密度（PSD）—— 感知每个变量的频谱能量分布
2. 交叉谱密度（CSD）特征值 —— 感知变量间的频域相关性
3. 频谱统计量 —— 感知整体频谱形态（质心、带宽、熵等）

设计原则：
- 输入：[B, S, D]  batch × seq_len × n_features
- 输出：[B, F]     batch × feature_dim
- 全可微（除了排序/统计量），方便后续路由器端到端训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


def compute_psd(x: torch.Tensor, n_fft: int = None,
                hop_length: int = None) -> torch.Tensor:
    """
    计算多变量功率谱密度（PSD）
    Args:
        x: [B, S, D]  batch × seq_len × n_features
        n_fft: FFT 窗口大小，默认 = seq_len
        hop_length: 不使用（单窗口 PSD）
    Returns:
        psd: [B, K, D]  batch × freq_bins × n_features
    """
    B, S, D = x.shape
    if n_fft is None:
        n_fft = S

    x_t = x.permute(0, 2, 1)  # [B, D, S]
    x_fft = torch.fft.rfft(x_t, n=n_fft, dim=-1)  # [B, D, K]
    psd = (x_fft.real ** 2 + x_fft.imag ** 2)  # [B, D, K]
    psd = psd.permute(0, 2, 1)  # [B, K, D]
    return psd


def compute_cross_spectral_eigs(x: torch.Tensor,
                                 n_fft: int = None) -> torch.Tensor:
    """
    计算交叉谱密度矩阵的特征值（捕获变量间频域相关性）
    对每个频点，计算 D×D 交叉谱矩阵，取其 top-k 特征值
    Args:
        x: [B, S, D]
    Returns:
        eigs: [B, K, top_k]  每个频点的特征值
    """
    B, S, D = x.shape
    if n_fft is None:
        n_fft = S
    K = n_fft // 2 + 1

    x_t = x.permute(0, 2, 1)  # [B, D, S]
    x_fft = torch.fft.rfft(x_t, n=n_fft, dim=-1)  # [B, D, K]

    top_k = min(D, 4)  # 取 top-4 特征值，控制维度
    eigs_list = []

    for k in range(K):
        csd_k = x_fft[:, :, k].unsqueeze(-1) * x_fft[:, :, k].unsqueeze(-2).conj()
        csd_k = csd_k.real  # [B, D, D]
        eigvals = torch.linalg.eigvalsh(csd_k)  # [B, D]
        eigvals = torch.flip(eigvals, dims=[-1])[:, :top_k]  # 取最大的 top_k 个
        eigs_list.append(eigvals)

    eigs = torch.stack(eigs_list, dim=1)  # [B, K, top_k]
    return eigs


def compute_spectral_stats(psd: torch.Tensor) -> torch.Tensor:
    """
    从 PSD 计算全局频谱统计量（每个变量一组统计量）
    Args:
        psd: [B, K, D]
    Returns:
        stats: [B, D * n_stats]
    """
    B, K, D = psd.shape

    freq_axis = torch.linspace(0, 0.5, K, device=psd.device).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]

    psd_norm = psd / (psd.sum(dim=1, keepdim=True) + 1e-8)  # [B, K, D]

    centroid = (freq_axis * psd_norm).sum(dim=1)  # [B, D]  频谱质心

    variance = ((freq_axis - centroid.unsqueeze(1)) ** 2 * psd_norm).sum(dim=1)  # [B, D]
    bandwidth = torch.sqrt(variance + 1e-8)  # [B, D]  频谱带宽

    entropy = -(psd_norm * torch.log(psd_norm + 1e-8)).sum(dim=1)  # [B, D]  频谱熵
    max_entropy = torch.log(torch.tensor(K, dtype=torch.float32, device=psd.device))
    entropy_norm = entropy / (max_entropy + 1e-8)

    rolloff_idx = torch.cumsum(psd_norm, dim=1)  # [B, K, D]
    rolloff_thresh = 0.85
    rolloff_mask = (rolloff_idx >= rolloff_thresh).float()
    rolloff_pos = rolloff_mask.argmax(dim=1).float() / K  # [B, D]  滚降频率

    stats = torch.cat([centroid, bandwidth, entropy_norm, rolloff_pos], dim=-1)  # [B, D*4]
    return stats


class MultiVariateSpectralExtractor(nn.Module):
    """
    多变量频谱特征提取器
    整合 PSD + 交叉谱特征值 + 统计量，输出压缩后的向量
    """

    def __init__(self, n_features: int, seq_len: int = 96,
                 output_dim: int = 128, use_csd: bool = True,
                 use_stats: bool = True):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_fft = seq_len
        self.K = seq_len // 2 + 1
        self.use_csd = use_csd
        self.use_stats = use_stats
        self.output_dim = output_dim

        raw_psd_dim = self.K * n_features
        csd_dim = self.K * 4 if use_csd else 0
        stats_dim = n_features * 4 if use_stats else 0

        total_raw = raw_psd_dim + csd_dim + stats_dim

        self.compressor = nn.Sequential(
            nn.Linear(total_raw, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def extract_raw(self, x: torch.Tensor) -> torch.Tensor:
        """返回 compressor 之前的原始频谱特征（PSD log + CSD eig log + stats）。

        特征只依赖输入 x，与 compressor / router_head 的参数无关，因此在路由器
        多 epoch 训练时可以一次算好、反复使用（见 `RouterCache.raw_feat`）。

        Args:
            x: [B, S, D]
        Returns:
            raw_feat: [B, total_raw]，total_raw = K*D (+ K*4) (+ D*4)
        """
        B = x.shape[0]

        psd = compute_psd(x, self.n_fft)  # [B, K, D]
        psd_log = torch.log1p(psd)  # 对数压缩
        psd_flat = psd_log.reshape(B, -1)  # [B, K*D]

        feat_list = [psd_flat]

        if self.use_csd:
            csd_eigs = compute_cross_spectral_eigs(x, self.n_fft)  # [B, K, 4]
            csd_log = torch.log1p(csd_eigs)
            csd_flat = csd_log.reshape(B, -1)  # [B, K*4]
            feat_list.append(csd_flat)

        if self.use_stats:
            stats = compute_spectral_stats(psd)  # [B, D*4]
            feat_list.append(stats)

        return torch.cat(feat_list, dim=-1)  # [B, total_raw]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
        Returns:
            feat: [B, output_dim]
        """
        return self.compressor(self.extract_raw(x))
