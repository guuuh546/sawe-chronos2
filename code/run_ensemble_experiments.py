"""
频谱感知动态集成 —— 完整实验脚本
==================================
包含：
E1: 主实验 —— 动态集成 vs 单一方法 vs 静态集成
E2: 消融实验 —— 不同特征组合、不同基模型组合
E3: Early-Exit 效率分析
E4: 跨数据集零样本泛化
E5: 可解释性可视化

用法：
    python run_ensemble_experiments.py --e1     # 主实验
    python run_ensemble_experiments.py --e3     # Early-Exit 分析
    python run_ensemble_experiments.py --e4     # 跨数据集泛化
    python run_ensemble_experiments.py --e5     # 可解释性可视化
    python run_ensemble_experiments.py --all    # 全部
"""
import argparse
import json
import os
import sys
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import build_backbone, get_target_modules
from src.model_injection import inject_lora_to_model, compute_total_spectral_reg
from src.baselines import (
    inject_dense_spectral_dw,
    inject_fourier_ft_faithful,
    inject_loft_llm,
)
from src.datasets import get_dataloaders_for_experiment
from src.trainer import Trainer
from src.ensemble import SpectrumRouter, SpectrumAwareEnsemble, train_router
from src.ensemble_analysis import collect_router_decisions, generate_all_plots, plot_cross_dataset_weight_comparison


# 基础模型选择："chronos2"（Amazon Chronos-2）或 "timesfm"（Google TimesFM）
BACKBONE = os.environ.get("TSFM_BACKBONE", "chronos2")
# 设备自动检测：可用 GPU 时默认 cuda，否则 cpu（可用环境变量 DEVICE 或 --device 覆盖）
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# 基础模型较大（Chronos-2 113M / TimesFM 494M）；8GB 显存下 TimesFM 实测 batch 16 峰值 5.4GB，
# 但两个基模型常驻 + 训练时接近上限，故 TimesFM 用 batch 8
BATCH_SIZE = 8 if BACKBONE == "timesfm" else 32
# Freq-LoRA 频谱门控的频点数：与 E1 对齐（seq_len=48 -> 48//2+1=25，min(65, 25)=25）
NUM_FREQ_BINS = min(65, 48 // 2 + 1)
# E2 消融的序列长度：与 E1/SADE 主实验对齐
SEQ_LEN = 48
PRED_LEN = 24
# 基模型训练轮数与早停耐心（可由 --epochs / --patience 覆盖；大基础模型可据此降低）
NUM_EPOCHS = 20
ES_PATIENCE = 5


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_base_model(input_dim, d_model=32, seq_len=48, pred_len=24):
    return build_backbone(BACKBONE, input_dim, seq_len, pred_len)


def train_single_model(method_name, base_model, train_loader, val_loader,
                      input_dim, device, seed, config=None):
    """训练单个基模型并返回训练好的模型和最佳验证 MSE"""
    set_seed(seed)
    model = build_base_model(input_dim).to(device)
    target_modules = get_target_modules(BACKBONE)

    if method_name == "vanilla_lora":
        model = inject_lora_to_model(
            model, rank=16, alpha=1.0,
            use_gating=False, use_spectral_reg=False,
            target_modules=target_modules
        )
        lr = 1e-3
    elif method_name == "freq_lora":
        model = inject_lora_to_model(
            model, rank=16, alpha=1.0,
            use_gating=True, use_spectral_reg=True,
            num_freq_bins=NUM_FREQ_BINS, lambda_l1=0.0001,
            target_modules=target_modules
        )
        lr = 1e-3
    elif method_name == "fourier_ft":
        # Gao et al. (ICML 2024) 忠实 FourierFT：n=1000, α=150, seed=777
        model = inject_fourier_ft_faithful(
            model, n_frequency=1000, scaling=150.0, random_loc_seed=777,
            target_modules=target_modules
        )
        lr = 1e-3
    elif method_name == "dense_spectral":
        # 附录消融：全频带稠密频谱 ΔW（旧实现，不是 FourierFT）
        model = inject_dense_spectral_dw(
            model, keep_ratio=0.5,
            target_modules=target_modules
        )
        lr = 1e-3
    elif method_name == "loft_llm":
        model = inject_loft_llm(
            model, rank=16, alpha=1.0, keep_ratio=0.5,
            target_modules=target_modules
        )
        lr = 1e-3
    elif method_name == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        lr = 1e-4
    elif method_name == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if any(h in n for h in ("output_head", "input_proj", "pred_head", "output_proj")):
                p.requires_grad = True
        lr = 1e-3
    else:
        raise ValueError(f"Unknown method: {method_name}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    use_reg = (method_name == "freq_lora")
    trainer = Trainer(
        model, optimizer, device=device,
        use_spectral_reg=use_reg, lambda_l1=0.0001
    )

    result = trainer.train(
        train_loader, val_loader,
        num_epochs=NUM_EPOCHS, early_stopping_patience=ES_PATIENCE
    )

    return model, result["best_val_mse"]


def evaluate_model(model, test_loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            total_mse += ((preds - y) ** 2).sum().item()
            total_mae += torch.abs(preds - y).sum().item()
            n += y.numel()
    return {"mse": total_mse / n, "mae": total_mae / n}


def run_e1_main(dataset_name, base_methods, ensemble_methods,
                seeds, device, output_dir):
    """E1: 主实验 —— 动态集成 vs 单一方法"""
    print(f"\n{'='*60}")
    print(f"E1 主实验: {dataset_name}")
    print(f"{'='*60}")

    all_results = {}

    dataloaders = get_dataloaders_for_experiment(
        dataset_name, batch_size=BATCH_SIZE, seq_len=48, pred_len=24
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    sample_x, sample_y = next(iter(train_loader))
    input_dim = sample_x.shape[-1]

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        seed_key = f"seed_{seed}"
        all_results[seed_key] = {}

        trained_models = {}

        # 训练基模型
        for method in base_methods:
            print(f"  训练 {method}...")
            model, best_val = train_single_model(
                method, None, train_loader, val_loader,
                input_dim, device, seed
            )
            test_metrics = evaluate_model(model, test_loader, device)
            trained_models[method] = model
            all_results[seed_key][method] = test_metrics
            print(f"    Test MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 缓解多模型常驻时的显存碎片

        # 静态集成（等权重）
        if "static_ensemble" in ensemble_methods:
            print(f"  静态集成（等权重）...")
            models_list = [trained_models[m] for m in base_methods]
            from src.ensemble import SpectrumAwareEnsemble, SpectrumRouter
            dummy_router = SpectrumRouter(
                n_features=input_dim, seq_len=48,
                num_models=len(base_methods)
            )
            static_ens = SpectrumAwareEnsemble(models_list, dummy_router)
            static_ens.eval()
            with torch.no_grad():
                all_preds = []
                all_targets = []
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    preds_list = [m(x) for m in models_list]
                    pred = sum(preds_list) / len(preds_list)
                    all_preds.append(pred)
                    all_targets.append(y)
                all_preds = torch.cat(all_preds, dim=0)
                all_targets = torch.cat(all_targets, dim=0)
                mse = ((all_preds - all_targets) ** 2).mean().item()
                mae = torch.abs(all_preds - all_targets).mean().item()
            all_results[seed_key]["static_ensemble"] = {"mse": mse, "mae": mae}
            print(f"    Test MSE: {mse:.4f}, MAE: {mae:.4f}")

        # 动态集成（频谱感知）
        if "dynamic_ensemble" in ensemble_methods:
            print(f"  动态集成（频谱感知）...")
            models_list = [trained_models[m] for m in base_methods]
            router = SpectrumRouter(
                n_features=input_dim, seq_len=48,
                num_models=len(base_methods),
                hidden_dim=128,
                use_csd=(input_dim <= 64), use_stats=True
            ).to(device)
            dynamic_ens = SpectrumAwareEnsemble(
                models_list, router, early_exit_threshold=0.9
            ).to(device)

            train_router(dynamic_ens, train_loader, val_loader,
                        lr=1e-3, num_epochs=20, device=device)

            test_metrics = evaluate_model(dynamic_ens, test_loader, device)
            all_results[seed_key]["dynamic_ensemble"] = test_metrics
            print(f"    Test MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")

            # 保存路由器决策用于可视化
            if seed == seeds[0]:
                try:
                    decisions = collect_router_decisions(
                        dynamic_ens, test_loader, device=device
                    )
                    vis_dir = os.path.join(output_dir, f"e1_{dataset_name}_visualization")
                    generate_all_plots(decisions, base_methods, vis_dir)
                except Exception as e:
                    print(f"    可视化跳过（缺少依赖或出错）: {e}")

    # 汇总
    summary = {}
    for method in base_methods + ensemble_methods:
        mses = [all_results[f"seed_{s}"][method]["mse"] for s in seeds]
        maes = [all_results[f"seed_{s}"][method]["mae"] for s in seeds]
        summary[method] = {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
        }

    print(f"\n--- {dataset_name} 汇总 ---")
    for method, metrics in summary.items():
        print(f"  {method:20s}: MSE={metrics['mse_mean']:.4f}±{metrics['mse_std']:.4f}, "
              f"MAE={metrics['mae_mean']:.4f}±{metrics['mae_std']:.4f}")

    return all_results, summary


def run_e3_early_exit(dataset_name, base_methods, seeds, device, output_dir):
    """E3: Early-Exit 效率分析"""
    print(f"\n{'='*60}")
    print(f"E3 Early-Exit 分析: {dataset_name}")
    print(f"{'='*60}")

    dataloaders = get_dataloaders_for_experiment(
        dataset_name, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, pred_len=PRED_LEN
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    sample_x, sample_y = next(iter(train_loader))
    input_dim = sample_x.shape[-1]

    seed = seeds[0]
    trained_models = {}
    for method in base_methods:
        print(f"  训练 {method}...")
        model, _ = train_single_model(
            method, None, train_loader, val_loader,
            input_dim, device, seed
        )
        trained_models[method] = model

    models_list = [trained_models[m] for m in base_methods]
    router = SpectrumRouter(
        n_features=input_dim, seq_len=SEQ_LEN,
        num_models=len(base_methods)
    ).to(device)
    ensemble = SpectrumAwareEnsemble(
        models_list, router, early_exit_threshold=0.9
    ).to(device)

    train_router(ensemble, train_loader, val_loader,
                lr=1e-3, num_epochs=20, device=device)

    # 阈值扫描：0.35–0.45 用于刻画路由器权重实际能达到的置信区间
    thresholds = [0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    results = {}

    max_weight_list = []
    weight_list = []

    ensemble.eval()
    with torch.no_grad():
        for thresh in thresholds:
            ensemble.early_exit_threshold = thresh
            total_mse = 0.0
            total_models = 0.0
            n_batches = 0
            exit_count = 0
            total_samples = 0
            # 路由器权重与阈值无关，只需在首个阈值时统计一次
            collect_stats = (thresh == thresholds[0])

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred, info = ensemble.forward_with_early_exit(x)
                total_mse += ((pred - y) ** 2).mean().item()
                total_models += info["avg_models_used"]
                exit_count += info["exit_ratio"] * x.shape[0]
                total_samples += x.shape[0]
                n_batches += 1
                if collect_stats:
                    max_weight_list.append(info["max_weight"])
                    weight_list.append(info["weights"])

            avg_mse = total_mse / n_batches
            avg_models = total_models / n_batches
            exit_ratio = exit_count / total_samples

            results[str(thresh)] = {
                "mse": avg_mse,
                "avg_models": avg_models,
                "exit_ratio": exit_ratio,
            }
            print(f"  Threshold={thresh:.2f}: MSE={avg_mse:.4f}, "
                  f"Avg Models={avg_models:.2f}, Exit={exit_ratio:.2%}", flush=True)

    # 路由器置信度统计：直接解释早退率为何为 0
    max_w = np.concatenate(max_weight_list)
    w_all = np.concatenate(weight_list, axis=0)
    weight_stats = {
        "n_samples": int(max_w.shape[0]),
        "max_weight_mean": float(max_w.mean()),
        "max_weight_p50": float(np.percentile(max_w, 50)),
        "max_weight_p90": float(np.percentile(max_w, 90)),
        "max_weight_p99": float(np.percentile(max_w, 99)),
        "max_weight_max": float(max_w.max()),
        "per_model_mean_weight": {
            m: float(w_all[:, i].mean()) for i, m in enumerate(base_methods)
        },
    }
    results["router_weight_stats"] = weight_stats
    print(f"  路由器 max weight: mean={weight_stats['max_weight_mean']:.4f}, "
          f"p50={weight_stats['max_weight_p50']:.4f}, "
          f"p90={weight_stats['max_weight_p90']:.4f}, "
          f"p99={weight_stats['max_weight_p99']:.4f}, "
          f"max={weight_stats['max_weight_max']:.4f}", flush=True)
    print(f"  各成员平均权重: {weight_stats['per_model_mean_weight']}", flush=True)

    save_path = os.path.join(output_dir, f"e3_early_exit_{dataset_name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 {save_path}", flush=True)

    return results


def run_e4_cross_dataset(source_dataset, target_datasets, base_methods,
                        seeds, device, output_dir):
    """E4: 跨数据集零样本泛化实验"""
    print(f"\n{'='*60}")
    print(f"E4 跨数据集泛化: {source_dataset} → ...")
    print(f"{'='*60}")

    seed = seeds[0]

    # 在源数据集上训练基模型和路由器
    print(f"\n在源数据集 {source_dataset} 上训练...")
    src_dls = get_dataloaders_for_experiment(
        source_dataset, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, pred_len=PRED_LEN
    )
    src_train = src_dls['train']
    src_val = src_dls['val']
    src_test = src_dls['test']
    sx, _ = next(iter(src_train))
    src_dim = sx.shape[-1]

    src_models = {}
    for method in base_methods:
        print(f"  训练 {method}...")
        model, _ = train_single_model(
            method, None, src_train, src_val, src_dim, device, seed
        )
        src_models[method] = model

    models_list = [src_models[m] for m in base_methods]
    router = SpectrumRouter(
        n_features=src_dim, seq_len=SEQ_LEN, num_models=len(base_methods)
    ).to(device)
    ensemble = SpectrumAwareEnsemble(models_list, router).to(device)
    train_router(ensemble, src_train, src_val,
                lr=1e-3, num_epochs=20, device=device)

    # 收集源数据集决策
    src_decisions = collect_router_decisions(ensemble, src_test, device)

    all_decisions = {source_dataset: src_decisions}
    results = {}

    # 在目标数据集上零样本测试
    for target in target_datasets:
        print(f"\n零样本测试 → {target}")
        try:
            tgt_dls = get_dataloaders_for_experiment(
                target, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, pred_len=PRED_LEN
            )
            tgt_train = tgt_dls['train']
            tgt_val = tgt_dls['val']
            tgt_test = tgt_dls['test']
            tx, _ = next(iter(tgt_train))
            tgt_dim = tx.shape[-1]

            # 注意：如果特征维度不同，路由器可能需要适配
            if tgt_dim != src_dim:
                print(f"  特征维度不同 ({src_dim} → {tgt_dim})，跳过（需要适配路由器输入层...")
                # 简化处理：重新初始化路由器的输入投影层
                old_extractor = ensemble.router.spectral_extractor
                # 这里为了简化，直接跳过或使用相同维度的数据集
                # 实际论文中可以用一个线性层做维度映射
                continue

            tgt_metrics = evaluate_model(ensemble, tgt_test, device)
            results[target] = tgt_metrics
            print(f"  Zero-shot MSE: {tgt_metrics['mse']:.4f}, MAE: {tgt_metrics['mae']:.4f}")

            tgt_decisions = collect_router_decisions(ensemble, tgt_test, device)
            all_decisions[target] = tgt_decisions

        except Exception as e:
            print(f"  跳过 {target}: {e}")

    # 画跨数据集权重对比图
    if len(all_decisions) > 1:
        vis_path = os.path.join(output_dir, "e4_cross_dataset_comparison.png")
        plot_cross_dataset_weight_comparison(
            all_decisions, base_methods, vis_path
        )

    save_path = os.path.join(output_dir, f"e4_cross_dataset_{source_dataset}.json")
    with open(save_path, "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print(f"\n结果已保存到 {save_path}")

    return results


def run_e2_ablation(dataset_name, base_methods, seeds, device, output_dir):
    """E2: 消融实验 —— 特征组合 + 基模型组合"""
    print(f"\n{'='*60}")
    print(f"E2 消融实验: {dataset_name}")
    print(f"{'='*60}")

    results = {}

    dataloaders = get_dataloaders_for_experiment(
        dataset_name, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, pred_len=PRED_LEN
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[-1]

    seed = seeds[0]
    set_seed(seed)

    # 先训练所有基模型（共享给各消融配置）
    print(f"\n--- 训练基模型 (seed={seed}) ---")
    trained_models = {}
    for method in base_methods:
        print(f"  训练 {method}...")
        model, _ = train_single_model(
            method, None, train_loader, val_loader,
            input_dim, device, seed
        )
        trained_models[method] = model

    # 消融 1: 不同特征组合
    print(f"\n--- 消融 1: 特征组合 ---")
    feature_configs = {
        "PSD_only": {"use_csd": False, "use_stats": False},
        "PSD+CSD": {"use_csd": True, "use_stats": False},
        "PSD+Stats": {"use_csd": False, "use_stats": True},
        "Full(PSD+CSD+Stats)": {"use_csd": True, "use_stats": True},
    }

    models_list = [trained_models[m] for m in base_methods]
    for cfg_name, cfg in feature_configs.items():
        print(f"  {cfg_name}...")
        router = SpectrumRouter(
            n_features=input_dim, seq_len=SEQ_LEN,
            num_models=len(base_methods),
            hidden_dim=128, **cfg
        ).to(device)
        ens = SpectrumAwareEnsemble(models_list, router).to(device)
        train_router(ens, train_loader, val_loader, lr=1e-3, num_epochs=20, device=device)
        metrics = evaluate_model(ens, test_loader, device)
        results[f"feature_{cfg_name}"] = metrics
        print(f"    MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")

    # 消融 2: 不同基模型组合
    print(f"\n--- 消融 2: 基模型组合 ---")
    model_combo_configs = {
        "All_3": base_methods,
        "No_FourierFT": [m for m in base_methods if m != "fourier_ft"],
        "No_FreqLoRA": [m for m in base_methods if m != "freq_lora"],
        "No_VanillaLoRA": [m for m in base_methods if m != "vanilla_lora"],
    }

    for cfg_name, combo_methods in model_combo_configs.items():
        if len(combo_methods) < 2:
            print(f"  {cfg_name}: 跳过（模型数 < 2）")
            continue
        print(f"  {cfg_name} ({len(combo_methods)} 模型)...")
        combo_models = [trained_models[m] for m in combo_methods]
        router = SpectrumRouter(
            n_features=input_dim, seq_len=SEQ_LEN,
            num_models=len(combo_methods),
            hidden_dim=128, use_csd=True, use_stats=True
        ).to(device)
        ens = SpectrumAwareEnsemble(combo_models, router).to(device)
        train_router(ens, train_loader, val_loader, lr=1e-3, num_epochs=20, device=device)
        metrics = evaluate_model(ens, test_loader, device)
        results[f"combo_{cfg_name}"] = metrics
        print(f"    MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")

    # 保存
    save_path = os.path.join(output_dir, f"e2_ablation_{dataset_name}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nE2 消融结果已保存到 {save_path}")
    return results


def main():
    global NUM_EPOCHS, ES_PATIENCE

    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", action="store_true", help="运行主实验")
    parser.add_argument("--e2", action="store_true", help="运行消融实验")
    parser.add_argument("--e3", action="store_true", help="运行 Early-Exit 实验")
    parser.add_argument("--e4", action="store_true", help="运行跨数据集泛化实验")
    parser.add_argument("--e5", action="store_true", help="运行可解释性可视化")
    parser.add_argument("--all", action="store_true", help="运行所有实验")
    parser.add_argument("--datasets", nargs="+",
                        default=["ETTh1", "Exchange", "Weather"],
                        help="数据集列表")
    parser.add_argument("--no_ensemble", action="store_true",
                        help="只跑单一方法（跳过静态/动态集成）；TimesFM 等大模型在 8GB 显存下的必需选项")
    parser.add_argument("--methods", nargs="+",
                        default=["vanilla_lora", "fourier_ft", "freq_lora"],
                        help="基模型方法列表")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="随机种子")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="基模型训练轮数（默认 20）")
    parser.add_argument("--patience", type=int, default=ES_PATIENCE,
                        help="早停耐心（默认 5）")
    parser.add_argument("--device", default=DEVICE, help="设备")
    parser.add_argument("--output_dir", default="outputs/ensemble_experiments",
                        help="输出目录")
    args = parser.parse_args()

    NUM_EPOCHS = args.epochs
    ES_PATIENCE = args.patience

    if args.all:
        args.e1 = args.e2 = args.e3 = args.e4 = args.e5 = True

    if not any([args.e1, args.e2, args.e3, args.e4, args.e5]):
        args.e1 = True  # 默认跑 E1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")

    ensemble_methods = [] if args.no_ensemble else ["static_ensemble", "dynamic_ensemble"]

    all_summaries = {}

    if args.e1:
        for ds in args.datasets:
            try:
                _, summary = run_e1_main(
                    ds, args.methods, ensemble_methods,
                    args.seeds, device, output_dir
                )
                all_summaries[ds] = summary
            except Exception as e:
                print(f"E1 {ds} 失败: {e}")
                import traceback
                traceback.print_exc()
            # 增量保存：每个数据集完成后立即落盘，避免中途崩溃丢失已跑完的结果
            with open(os.path.join(output_dir, "e1_overall_summary.json"), "w") as f:
                json.dump(all_summaries, f, indent=2)

        print(f"\nE1 总体汇总已保存")

    if args.e2:
        for ds in args.datasets:
            try:
                run_e2_ablation(ds, args.methods, args.seeds, device, output_dir)
            except Exception as e:
                print(f"E2 {ds} 失败: {e}")
                import traceback
                traceback.print_exc()

    if args.e3:
        for ds in args.datasets:
            try:
                run_e3_early_exit(ds, args.methods, args.seeds, device, output_dir)
            except Exception as e:
                print(f"E3 {ds} 失败: {e}")

    if args.e4:
        if len(args.datasets) >= 2:
            source = args.datasets[0]
            targets = args.datasets[1:]
            try:
                run_e4_cross_dataset(source, targets, args.methods,
                                     args.seeds, device, output_dir)
            except Exception as e:
                print(f"E4 失败: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"所有实验完成！结果在: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
