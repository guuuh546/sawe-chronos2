"""
完整部署路径（先筛选、后组合）—— 决策与补算
================================================
目的：回答"按本文准则部署后，7 个数据集的完整流程结果如何"。

做法（关键：筛选决策只使用**验证集**，绝不使用测试集）：
  1) stage=val：对 7 数据集 x 3 种子，计算每个成员的验证集/测试集 MSE，
     并按阈值 theta（最差/最优 验证误差比）给出筛选决策。
  2) stage=screened：仅对"决策要求剔除成员"的 (数据集, 种子)，
     在缩减后的成员集上重训路由器并评测测试集（其余数据集无需重训，
     其完整流程结果 = 已报告的 SAWE 结果）。
  3) summary-only：打印决策表与补算结果。

用法：
    python -u _full_pipeline.py --stage val
    python -u _full_pipeline.py --stage screened --theta 1.2
    python -u _full_pipeline.py --summary-only
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_CACHE_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import run_ensemble_experiments as R
import _multiseed as M
from src.datasets import get_dataloaders_for_experiment

DATASETS = ["exchange_rate", "ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather"]
SEEDS = [42, 123, 456]
METHODS = M.METHODS
# 与 _multiseed 保持一致：默认旧目录；改稿重跑用 --out 指向独立目录
OUT = os.environ.get("REVISION_OUT", os.path.join("outputs", "multiseed"))

THETAS = [1.10, 1.20, 1.30, 1.50, float("inf")]

SUBSETS = {
    "All_3": METHODS,
    "No_FourierFT": [m for m in METHODS if m != "fourier_ft"],
    "No_FreqLoRA": [m for m in METHODS if m != "freq_lora"],
    "No_VanillaLoRA": [m for m in METHODS if m != "vanilla_lora"],
}
NAME_BY_SUBSET = {tuple(v): k for k, v in SUBSETS.items()}


def val_path(ds, seed):
    return os.path.join(OUT, f"fp_val_{ds}_seed{seed}.json")


def scr_path(ds, seed):
    return os.path.join(OUT, f"fp_screened_{ds}_seed{seed}.json")


def decide(val_mse, theta):
    """按验证集误差比决定剔除哪些成员（至少保留 2 个）。"""
    keep = list(METHODS)
    dropped = []
    if theta == float("inf"):
        return keep, dropped
    while len(keep) > 2:
        vals = {m: val_mse[m] for m in keep}
        worst = max(vals, key=vals.get)
        best = min(vals.values())
        if vals[worst] / max(best, 1e-12) > theta:
            keep.remove(worst)
            dropped.append(worst)
        else:
            break
    return keep, dropped


def stage_val(ds, seed):
    dls = get_dataloaders_for_experiment(
        ds, batch_size=R.BATCH_SIZE, seq_len=R.SEQ_LEN, pred_len=R.PRED_LEN)
    x0, _ = next(iter(dls["train"]))
    input_dim = x0.shape[-1]
    models = M.train_or_load_members(ds, seed, dls, input_dim)

    with torch.no_grad():
        pv = {m: M.predict(models[m], dls["val_dataset"]) for m in METHODS}
        pt = {m: M.predict(models[m], dls["test_dataset"]) for m in METHODS}
    Yv = M.collect_y(dls["val_dataset"])
    Yt = M.collect_y(dls["test_dataset"])

    val_mse = {m: float(((pv[m] - Yv) ** 2).mean()) for m in METHODS}
    test_mse = {m: float(((pt[m] - Yt) ** 2).mean()) for m in METHODS}

    # 等权集成（无需训练，可作为参考）
    static_all = float(((torch.stack([pt[m] for m in METHODS], 0).mean(0) - Yt) ** 2).mean())

    decisions = {}
    for th in THETAS:
        keep, dropped = decide(val_mse, th)
        decisions[str(th)] = {
            "theta": th,
            "subset": NAME_BY_SUBSET[tuple(keep)],
            "drop": dropped,
            "val_ratio": round(max(val_mse[m] for m in keep) /
                               max(min(val_mse[m] for m in keep), 1e-12), 4),
        }

    out = {
        "dataset": ds, "seed": seed, "input_dim": input_dim,
        "val_mse": {m: round(v, 6) for m, v in val_mse.items()},
        "test_mse": {m: round(v, 6) for m, v in test_mse.items()},
        "best_single_name": min(test_mse, key=test_mse.get),
        "best_single_mse": round(min(test_mse.values()), 6),
        "static_all3_mse": round(static_all, 6),
        "decisions": decisions,
    }
    for m in list(models.keys()):
        del models[m]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def dataset_decision(ds, theta):
    """数据集级决策：对可用的各种子验证 MSE 取平均后判定（论文协议）"""
    recs = []
    for s in SEEDS:
        p = val_path(ds, s)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                recs.append(json.load(f))
    if not recs:
        raise FileNotFoundError(f"{ds}: 缺少 fp_val_*.json，请先运行 --stage val")
    mean_val = {m: float(np.mean([r["val_mse"][m] for r in recs])) for m in METHODS}
    keep, dropped = decide(mean_val, theta)
    return NAME_BY_SUBSET[tuple(keep)], mean_val, dropped


def stage_screened(ds, seed, subset_name, theta):
    """在数据集级决策指定的子集上重训路由器（subset_name == All_3 则跳过）。"""
    if subset_name == "All_3":
        return None

    subset = SUBSETS[subset_name]
    with open(val_path(ds, seed), encoding="utf-8") as f:
        base = json.load(f)
    dls = get_dataloaders_for_experiment(
        ds, batch_size=R.BATCH_SIZE, seq_len=R.SEQ_LEN, pred_len=R.PRED_LEN)
    models = M.train_or_load_members(ds, seed, dls, base["input_dim"])

    router = R.SpectrumRouter(n_features=base["input_dim"], seq_len=R.SEQ_LEN,
                             num_models=len(subset), hidden_dim=128,
                             use_csd=True, use_stats=True).to(R.DEVICE)
    ens = R.SpectrumAwareEnsemble([models[m] for m in subset], router).to(R.DEVICE)
    R.train_router(ens, dls["train"], dls["val"], lr=1e-3,
                   num_epochs=20, device=R.DEVICE)
    met = R.evaluate_model(ens, dls["test"], R.DEVICE)

    out = {
        "dataset": ds, "seed": seed, "theta": theta,
        "subset": subset_name, "members": subset,
        "screened_sawe_mse": round(float(met["mse"]), 6),
        "screened_sawe_mae": round(float(met["mae"]), 6),
    }
    del ens, router
    for m in list(models.keys()):
        del models[m]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def save(obj, path):
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {path}", flush=True)


def ms(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "per_seed": [float(v) for v in a]}


def summary():
    # ---------- 数据集级决策（论文协议：多 seeds 平均） ----------
    print("\n" + "=" * 104)
    print("数据集级筛选决策（对 3 个种子的验证 MSE 取平均后判定；符合论文『多 seeds 一致地差』的准则）")
    print("=" * 104)
    print(f"{'数据集':<14}{'Vanilla':>10}{'Fourier':>10}{'FreqLoRA':>10}"
          f"{'ratio':>8}{'@1.1':>16}{'@1.2':>16}{'@1.3':>16}")
    print("-" * 104)
    ds_level = {}
    for ds in DATASETS:
        recs = []
        for s in SEEDS:
            p = val_path(ds, s)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    recs.append(json.load(f))
        if not recs:
            continue
        mean_val = {m: float(np.mean([r["val_mse"][m] for r in recs])) for m in METHODS}
        ratio = max(mean_val.values()) / min(mean_val.values())
        ds_level[ds] = {"mean_val": mean_val, "ratio": ratio, "n_seeds": len(recs)}
        cells = []
        for th in (1.1, 1.2, 1.3):
            keep, _ = decide(mean_val, th)
            cells.append(NAME_BY_SUBSET[tuple(keep)])
        print(f"{ds:<14}{mean_val['vanilla_lora']:>10.4f}{mean_val['fourier_ft']:>10.4f}"
              f"{mean_val['freq_lora']:>10.4f}{ratio:>8.2f}"
              f"{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}")

    # ---------- 逐种子决策（用于观察稳定性） ----------
    print("\n" + "=" * 104)
    print("逐种子决策（阈值 1.2；用于检查决策是否随种子漂移）")
    print("=" * 104)
    for ds in DATASETS:
        recs = []
        for s in SEEDS:
            p = val_path(ds, s)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    recs.append(json.load(f))
        if not recs:
            continue
        subs = [r["decisions"]["1.2"]["subset"] for r in recs]
        ratios = [r["decisions"]["1.2"]["val_ratio"] for r in recs]
        print(f"{ds:<14} seeds={[r['seed'] for r in recs]}  "
              f"decisions={subs}  ratios={[round(z, 2) for z in ratios]}")

    # ---------- 补算结果 ----------
    print("\n" + "=" * 104)
    print("筛选后的组合结果（仅列出需要剔除成员的数据集）")
    print("=" * 104)
    for ds in DATASETS:
        for s in SEEDS:
            p = scr_path(ds, s)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    r = json.load(f)
                print(f"{ds:<14} seed={s:<4} {r['subset']:<16} "
                      f"SAWE={r['screened_sawe_mse']:.4f}")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "screened"], default="val")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--theta", type=float, default=1.2)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--out", default=None,
                    help="输出/读取目录（须与 _multiseed 的 --out 一致）")
    args = ap.parse_args()

    if args.out:
        OUT = args.out
        M.OUT = args.out      # train_or_load_members / mem_path 依赖 M.OUT
    os.makedirs(OUT, exist_ok=True)
    print(f"[out] {OUT}", flush=True)
    if args.summary_only:
        summary()
        return

    datasets = args.datasets or DATASETS
    seeds = args.seeds or SEEDS

    for ds in datasets:
        dec_name = None
        if args.stage == "screened":
            dec_name, mean_val, dropped = dataset_decision(ds, args.theta)
            print(f"\n[decision] {ds}: subset={dec_name} theta={args.theta} "
                  f"mean_val={ {k: round(v, 4) for k, v in mean_val.items()} }",
                  flush=True)
        for seed in seeds:
            if args.stage == "val":
                p = val_path(ds, seed)
                if os.path.exists(p):
                    print(f"[skip] {ds} seed={seed}", flush=True)
                    continue
                print(f"\n{'=' * 60}\n[val] {ds} seed={seed}\n{'=' * 60}", flush=True)
                save(stage_val(ds, seed), p)
            else:
                p = scr_path(ds, seed)
                if os.path.exists(p):
                    print(f"[skip] {ds} seed={seed}", flush=True)
                    continue
                print(f"\n{'=' * 60}\n[screened] {ds} seed={seed}\n{'=' * 60}", flush=True)
                res = stage_screened(ds, seed, dec_name, args.theta)
                if res is None:
                    print(f"  [no-screen] {ds} seed={seed}: 决策为 All_3，无需重训", flush=True)
                else:
                    save(res, p)

    summary()


if __name__ == "__main__":
    main()
