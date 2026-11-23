try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np
import torch
from typing import List, Dict, Optional


def plot_spectral_gating_weights(
    gating_weights: np.ndarray,
    freq_bins: Optional[List[int]] = None,
    title: str = "Spectral Gating Weights Heatmap",
    save_path: Optional[str] = None,
):
    """Plot gating weights heatmap"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plotting.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    if freq_bins is None:
        freq_bins = [f"Bin {i}" for i in range(gating_weights.shape[1])]

    sns.heatmap(
        gating_weights,
        cmap="viridis",
        xticklabels=freq_bins,
        yticklabels=False,
        ax=ax,
        cbar=True,
    )

    ax.set_xlabel("Frequency Bins")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_frequency_preference_analysis(
    gating_weights_by_domain: Dict[str, np.ndarray],
    domain_names: Optional[List[str]] = None,
    title: str = "Frequency Preference Analysis by Domain",
    save_path: Optional[str] = None,
):
    """Plot frequency preference analysis for different domains"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plotting.")
        return

    if domain_names is None:
        domain_names = list(gating_weights_by_domain.keys())

    n_domains = len(domain_names)
    fig, axes = plt.subplots(n_domains, 1, figsize=(12, 4 * n_domains), sharex=True)
    if n_domains == 1:
        axes = [axes]

    for i, domain in enumerate(domain_names):
        weights = gating_weights_by_domain[domain]
        mean_weights = np.mean(weights, axis=0)
        std_weights = np.std(weights, axis=0)

        axes[i].plot(mean_weights, color="blue", linewidth=2, label="Mean Weight")
        axes[i].fill_between(
            range(len(mean_weights)),
            mean_weights - std_weights,
            mean_weights + std_weights,
            alpha=0.3,
            color="blue",
            label="±1 Std Dev",
        )
        axes[i].set_title(f"Frequency Preference - {domain} Domain")
        axes[i].set_ylabel("Gating Weight")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency Bins")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_progress(
    train_history: List[Dict],
    val_history: List[Dict],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
):
    """Plot training progress"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plotting.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot([h["loss"] for h in train_history], label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([h["mse"] for h in train_history], label="Train MSE")
    if val_history:
        axes[1].plot([h["mse"] for h in val_history], label="Val MSE", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("MSE Metric")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot([h["mae"] for h in train_history], label="Train MAE")
    if val_history:
        axes[2].plot([h["mae"] for h in val_history], label="Val MAE", linestyle="--")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE")
    axes[2].set_title("MAE Metric")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_gamma_distribution(
    gammas: np.ndarray,
    title: str = "Gating Weight γ Distribution",
    save_path: Optional[str] = None,
):
    """Plot distribution of gating weights gamma"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plotting.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(gammas.flatten(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("γ Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("γ Histogram")
    axes[0].grid(True, alpha=0.3)

    # Boxplot
    axes[1].boxplot(gammas.flatten())
    axes[1].set_ylabel("γ Value")
    axes[1].set_title("γ Boxplot")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_spectral_features_analysis(
    spectral_feats: np.ndarray,
    title: str = "Spectral Features Analysis",
    save_path: Optional[str] = None,
):
    """Plot analysis of spectral features"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Mean spectrum
    mean_spec = np.mean(spectral_feats, axis=0)
    axes[0, 0].plot(mean_spec, color='blue', linewidth=2)
    axes[0, 0].set_xlabel("Frequency Bins")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("Mean Spectrum")
    axes[0, 0].grid(True, alpha=0.3)

    # Spectrum heatmap
    sns.heatmap(
        spectral_feats[:min(100, len(spectral_feats))],
        cmap="viridis",
        ax=axes[0, 1],
        cbar=True,
        yticklabels=False
    )
    axes[0, 1].set_xlabel("Frequency Bins")
    axes[0, 1].set_title("Spectrum Heatmap (First 100 samples)")

    # Standard deviation
    std_spec = np.std(spectral_feats, axis=0)
    axes[1, 0].plot(std_spec, color='red', linewidth=2)
    axes[1, 0].set_xlabel("Frequency Bins")
    axes[1, 0].set_ylabel("Standard Deviation")
    axes[1, 0].set_title("Spectrum Standard Deviation")
    axes[1, 0].grid(True, alpha=0.3)

    # Frequency bin correlation
    if spectral_feats.shape[1] > 1:
        corr_matrix = np.corrcoef(spectral_feats.T)
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            ax=axes[1, 1],
            center=0,
            cbar=True
        )
        axes[1, 1].set_title("Frequency Bin Correlation Matrix")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_frequency_bands_analysis(
    gammas: np.ndarray,
    spectral_feats: np.ndarray,
    num_bands: int = 4,
    title: str = "Frequency Band Analysis",
    save_path: Optional[str] = None,
):
    """Divide frequency bins into bands and analyze"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping plotting.")
        return

    num_bins = spectral_feats.shape[1]
    bins_per_band = num_bins // num_bands

    # Calculate mean spectrum and mean gamma for each band
    band_names = []
    band_mean_spec = []
    band_mean_gamma = []

    for i in range(num_bands):
        start = i * bins_per_band
        end = (i + 1) * bins_per_band if i < num_bands - 1 else num_bins
        band_names.append(f"Band {i+1}\n({start}-{end})")
        band_mean_spec.append(np.mean(spectral_feats[:, start:end]))
        # Since gamma is a scalar, we find the corresponding relationship
        # Here we simply calculate the overall gamma mean
        band_mean_gamma.append(np.mean(gammas))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Frequency band spectrum
    axes[0].bar(range(num_bands), band_mean_spec, color='skyblue', edgecolor='black')
    axes[0].set_xticks(range(num_bands))
    axes[0].set_xticklabels(band_names)
    axes[0].set_ylabel("Mean Spectrum Amplitude")
    axes[0].set_title("Mean Spectrum by Frequency Band")
    axes[0].grid(True, alpha=0.3, axis='y')

    # Gamma distribution
    axes[1].violinplot(gammas.flatten(), showmeans=True, showmedians=True)
    axes[1].set_ylabel("γ Value")
    axes[1].set_title("Gating Weight γ Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
