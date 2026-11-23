"""频带划分（4.5 分带评估 / 分带报告）
=====================================

把 ``freq_contribution`` 输出的 K 个 bin（K = T//2 + 1）按频率从低到高
均分成 3 段：low / middle / high。

K=13（pred_len=24）时 edges = [0, 4, 9, 13]，即
    low    = bins 0..3
    middle = bins 4..8
    high   = bins 9..12

用法：
    from _band_report import make_bands
    masks, labels, edges = make_bands(13)
"""
import torch

BAND_LABELS = ("low", "middle", "high")


def make_bands(k: int):
    """把 K 个频率 bin 均分成 3 个互不重叠、完整覆盖的频带。

    Args:
        k: bin 数（= pred_len // 2 + 1）。

    Returns:
        masks: 3 个长度为 K 的 bool 张量（list）；
        labels: 与 masks 对应的标签 ``["low", "middle", "high"]``；
        edges: 4 个切分点 ``[0, e1, e2, K]``（K=13 -> [0, 4, 9, 13]）。
    """
    k = int(k)
    if k < 3:
        raise ValueError(f"make_bands 需要至少 3 个 bin，实际 K={k}")

    e1 = max(1, int(round(k / 3.0)))
    e2 = min(k - 1, max(e1 + 1, int(round(2.0 * k / 3.0))))
    edges = [0, e1, e2, k]

    masks = []
    for i in range(3):
        m = torch.zeros(k, dtype=torch.bool)
        m[edges[i]:edges[i + 1]] = True
        masks.append(m)
    return masks, list(BAND_LABELS), edges
