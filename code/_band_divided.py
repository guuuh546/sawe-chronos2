"""分带误差能量分解（4.5 分带评估的底层实现）
================================================

把逐元素平方误差按预测长度方向（dim=1）的频带拆开，得到每个频带对
整体 MSE 的贡献，且**严格满足** ``contrib.sum() == err.pow(2).mean()``。

口径（Parseval，单边谱）
------------------------
对每条残差 x[t]（t=0..T-1）做未归一化的 ``rfft`` 得 X[k]（k=0..T//2）：

    sum_t x[t]^2 = (1/T) * ( |X[0]|^2 + |X[T/2]|^2 + 2 * sum_{k=1}^{T/2-1} |X[k]|^2 )

即 DC（k=0）与 Nyquist（T 为偶数时的 k=T/2）权重为 1，其余频率权重为 2。
再除以元素总数 N*T*C，即得到"逐元素 MSE 尺度"的分带贡献：

    c[k] = w[k] * sum_{n,c} |X[k]|^2 / (N * T^2 * C)

于是 ``c.sum()`` 恰好等于 ``err.pow(2).mean()``（float64 下误差 < 1e-10）。

用法：
    from _band_divided import freq_contribution
    c = freq_contribution(pred - y)      # (N, T, C) -> (K,)
"""
import torch


def freq_contribution(err: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """把逐元素 MSE 分解到 dim 方向的各个频率 bin（单边谱，Parseval 加权）。

    Args:
        err: 残差张量，形状 ``(N, T, C)``（dim=1 为预测长度方向）。
        dim: 做 rfft 的维度，默认 1。

    Returns:
        形状 ``(K,)``、``K = T // 2 + 1`` 的 float64 张量，``sum() == err.pow(2).mean()``。
        计算全程使用 float64（分带占比是"小量相除"，float32 累加误差会污染百分比）。
    """
    if err.ndim != 3:
        raise ValueError(f"freq_contribution 期望 (N, T, C)，实际 shape={tuple(err.shape)}")
    if dim != 1:
        err = err.transpose(1, dim)

    n, t, c = err.shape
    if n == 0 or t == 0 or c == 0:
        raise ValueError(f"freq_contribution 收到空张量 shape={tuple(err.shape)}")

    spec = torch.fft.rfft(err.double(), dim=1)          # (N, K, C) complex128
    power = spec.real.pow(2) + spec.imag.pow(2)          # (N, K, C)
    k = spec.shape[1]

    # Parseval 单边加权：DC 与 Nyquist 权 1，其余权 2
    weight = torch.full((k,), 2.0, dtype=torch.float64, device=err.device)
    weight[0] = 1.0
    if t % 2 == 0:
        weight[-1] = 1.0

    contrib = power.sum(dim=(0, 2)) * weight / (n * t * t * c)
    return contrib
