"""
频谱感知动态集成 —— 可解释性分析工具
======================================
生成论文中需要的可视化图表：
1. 频谱-权重热力图（Router 决策可视化）
2. 不同数据特性下的权重分布
3. Early-Exit 触发比例分析
4. 跨数据集权重分布对比
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
import os

plt = None  # 懒加载 matplotlib


def _ensure_matplotlib():
    """懒加载 matplotlib，避免未安装时 import 报错"""
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module
        plt = plt_module
    return plt


def collect_router_decisions(ensemble, data_loader, device="cpu",
                              max_batches=20) -> Dict:
    """
    收集路由器的决策数据，用于后续分析和可视化
    """
    ensemble.eval()
    all_weights = []
    all_spectral_feats = []
    all_mse_per_model = []
    all_max_weights = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            weights = ensemble.router(x)  # [B, M]
            all_weights.append(weights.cpu().numpy())
            all_max_weights.append(weights.max(dim=-1).values.cpu().numpy())

            # 收集频谱特征
            spectral_feat = ensemble.router.spectral_extractor(x)
            all_spectral_feats.append(spectral_feat.cpu().numpy())

            # 收集每个模型的 MSE
            mses = []
            for model in ensemble.models:
                pred = model(x)
                mse = ((pred - y) ** 2).mean(dim=(1, 2))
                mses.append(mse.cpu().numpy())
            all_mse_per_model.append(np.stack(mses, axis=-1))

    result = {
        "weights": np.vstack(all_weights),          # [N, M]
        "max_weights": np.concatenate(all_max_weights),  # [N]
        "spectral_feats": np.vstack(all_spectral_feats), # [N, F]
        "mse_per_model": np.vstack(all_mse_per_model),   # [N, M]
    }
    return result


def plot_weight_distribution(decisions: Dict, model_names: List[str],
                              save_path: str):
    """画出每个基模型的权重分布直方图"""
    plt = _ensure_matplotlib()
    M = len(model_names)
    fig, axes = plt.subplots(1, M, figsize=(4 * M, 3), sharey=True)

    for i, name in enumerate(model_names):
        ax = axes[i] if M > 1 else axes
        w = decisions["weights"][:, i]
        ax.hist(w, bins=30, alpha=0.7, color=f"C{i}", edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("Weight")
        ax.set_xlim(0, 1)
        if i == 0:
            ax.set_ylabel("Count")
        ax.axvline(w.mean(), color="red", linestyle="--", label=f"Mean={w.mean():.2f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  权重分布 → {save_path}")


def plot_spectrum_weight_heatmap(decisions: Dict, model_names: List[str],
                                  save_path: str, n_bins: int = 10):
    """
    频谱-权重热力图（论文核心图）
    X轴：频谱质心分桶（低 → 高）
    Y轴：不同模型
    颜色：平均权重
    """
    from src.spectral_features import compute_psd

    plt = _ensure_matplotlib()

    weights = decisions["weights"]  # [N, M]
    feats = decisions["spectral_feats"]  # [N, F]

    # 用第一维作为"频谱复杂度"代理（也可以用质心等）
    # 这里直接用 feats 的 L2 范数作为复杂度代理
    complexity = np.linalg.norm(feats, axis=-1)  # [N]

    bin_edges = np.percentile(complexity, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(complexity.min(), complexity.max(), n_bins + 1)

    bin_ids = np.digitize(complexity, bin_edges[1:-1])

    M = weights.shape[1]
    n_bins_actual = len(bin_edges) - 1
    heatmap = np.zeros((M, n_bins_actual))

    for b in range(n_bins_actual):
        mask = bin_ids == b
        if mask.sum() > 0:
            heatmap[:, b] = weights[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(M))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Spectral Complexity (Low → High)")
    ax.set_title("Average Model Weight vs Spectral Complexity")

    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(n_bins_actual)]
    step = max(1, n_bins_actual // 5)
    ax.set_xticks(range(0, n_bins_actual, step))
    ax.set_xticklabels([f"Bin {i}" for i in range(0, n_bins_actual, step)], rotation=45)

    plt.colorbar(im, ax=ax, label="Avg Weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  频谱-权重热力图 → {save_path}")


def plot_early_exit_analysis(decisions: Dict, thresholds: List[float],
                              save_path: str):
    """画出不同阈值下的 Early-Exit 比例和平均使用模型数"""
    plt = _ensure_matplotlib()
    max_w = decisions["max_weights"]  # [N]
    M = decisions["weights"].shape[1]

    exit_ratios = []
    avg_models = []

    for thresh in thresholds:
        exit_mask = max_w >= thresh
        exit_ratio = exit_mask.mean()
        n_models = np.where(exit_mask, 1, M)
        avg_n = n_models.mean()
        exit_ratios.append(exit_ratio)
        avg_models.append(avg_n)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    color1 = "tab:blue"
    ax1.set_xlabel("Early-Exit Threshold")
    ax1.set_ylabel("Exit Ratio", color=color1)
    ax1.plot(thresholds, exit_ratios, "o-", color=color1, label="Exit Ratio")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Avg Models Used", color=color2)
    ax2.plot(thresholds, avg_models, "s--", color=color2, label="Avg Models")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.5, M + 0.5)
    ax2.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(M, color="gray", linestyle=":", alpha=0.5)

    plt.title("Early-Exit Efficiency Analysis")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Early-Exit 分析 → {save_path}")


def plot_cross_dataset_weight_comparison(decisions_dict: Dict[str, Dict],
                                          model_names: List[str],
                                          save_path: str):
    """
    跨数据集权重分布对比（用于跨数据集泛化实验）
    decisions_dict: {dataset_name: decisions_dict}
    """
    plt = _ensure_matplotlib()
    datasets = list(decisions_dict.keys())
    M = len(model_names)

    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ds_idx, ds_name in enumerate(datasets):
        ax = axes[ds_idx]
        dec = decisions_dict[ds_name]
        w_mean = dec["weights"].mean(axis=0)
        w_std = dec["weights"].std(axis=0)

        bars = ax.bar(range(M), w_mean, yerr=w_std, capsize=5,
                      color=[f"C{i}" for i in range(M)], alpha=0.8)
        ax.set_xticks(range(M))
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_title(ds_name)
        ax.set_ylim(0, 1)
        if ds_idx == 0:
            ax.set_ylabel("Average Weight")

    plt.suptitle("Cross-Dataset Weight Distribution Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  跨数据集对比 → {save_path}")


def generate_all_plots(decisions: Dict, model_names: List[str],
                        output_dir: str):
    """一键生成所有可解释性图表"""
    os.makedirs(output_dir, exist_ok=True)

    plot_weight_distribution(decisions, model_names,
                             os.path.join(output_dir, "weight_distribution.png"))
    plot_spectrum_weight_heatmap(decisions, model_names,
                                 os.path.join(output_dir, "spectrum_weight_heatmap.png"))
    plot_early_exit_analysis(
        decisions,
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        save_path=os.path.join(output_dir, "early_exit_analysis.png")
    )

    print(f"\n所有图表已保存到: {output_dir}")
