"""校准口径对照（回答"分划间幅度漂移"）

对每个已有成员缓存的数据集 × 种子，计算：
  1) 三个划分上 y 的标准差（检查幅度漂移）与预测的标准差
  2) 各方法在验证集上拟合的最优标量 c* = <p_val, y_val> / <p_val, p_val>
  3) 各方法在"未校准 / 标量校准"两种口径下的测试集 MSE

用于说明：后处理校准带来的收益与"选哪个成员/如何组合"正交，且量级更大。
输出：outputs/multiseed/calib_protocol.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_CACHE_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_ensemble_experiments as R
import _multiseed as M
from _multiseed import build_injected, mem_path
from src.datasets import get_dataloaders_for_experiment

METHODS = ["vanilla_lora", "fourier_ft", "freq_lora"]
# 与 _multiseed 保持一致：默认旧目录；改稿重跑用 --out 指向独立目录
OUT = os.environ.get("REVISION_OUT", os.path.join("outputs", "multiseed"))
SEEDS = [42, 123, 456]


@torch.no_grad()
def preds_of(model, ds, batch=128):
    ps = []
    for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
        ps.append(model(x.to(R.DEVICE)).cpu().reshape(-1).double())
    return torch.cat(ps)


@torch.no_grad()
def ys_of(ds, batch=128):
    return torch.cat([y.reshape(-1).double() for _, y in
                      DataLoader(ds, batch_size=batch, shuffle=False)])


def stats(p, y):
    c = float((p @ y) / (p @ p))
    return {"c_star": round(c, 4),
            "std_ratio": round(float(p.std() / y.std()), 4),
            "mse_raw": round(float(((p - y) ** 2).mean()), 6),
            "mse_cal": round(float(((c * p - y) ** 2).mean()), 6)}


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="成员缓存与结果目录（须与 _multiseed 的 --out 一致）")
    args = ap.parse_args()
    if args.out:
        OUT = args.out
        M.OUT = args.out      # mem_path / train_or_load_members 依赖 M.OUT
    os.makedirs(OUT, exist_ok=True)
    print(f"[out] {OUT}", flush=True)

    res = {}
    for ds in ["exchange_rate", "ETTh2"]:
        if not os.path.exists(mem_path(ds, SEEDS[0])):
            print(f"[skip] {ds}: 无成员缓存", flush=True)
            continue
        dls = get_dataloaders_for_experiment(ds, batch_size=R.BATCH_SIZE,
                                             seq_len=R.SEQ_LEN, pred_len=R.PRED_LEN)
        x0, _ = next(iter(dls["train"]))
        input_dim = x0.shape[-1]
        Y = {sp: ys_of(dls[f"{sp}_dataset"]) for sp in ["train", "val", "test"]}
        ds_entry = {"y_std": {sp: round(float(Y[sp].std()), 4) for sp in Y}, "seeds": {}}

        for seed in SEEDS:
            if not os.path.exists(mem_path(ds, seed)):
                continue
            ckpt = torch.load(mem_path(ds, seed), map_location=R.DEVICE)
            P = {}
            for m in METHODS:
                model = build_injected(m, input_dim, seed)
                sd = model.state_dict()
                sd.update({k: v.to(R.DEVICE) for k, v in ckpt[m].items()})
                model.load_state_dict(sd)
                model.eval()
                P[m] = {sp: preds_of(model, dls[f"{sp}_dataset"]) for sp in ["train", "val", "test"]}
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # 静态等权（成员均值）
            P["static_ensemble"] = {sp: torch.stack([P[m][sp] for m in METHODS], 0).mean(0)
                                    for sp in ["train", "val", "test"]}

            entry = {}
            for m in METHODS + ["static_ensemble"]:
                entry[m] = {
                    "train": stats(P[m]["train"], Y["train"]),
                    "val": stats(P[m]["val"], Y["val"]),
                    "test": stats(P[m]["test"], Y["test"]),
                }
                # 校准收益：用 val 拟合的 c* 作用到 test
                c = entry[m]["val"]["c_star"]
                raw = entry[m]["test"]["mse_raw"]
                cal = float(((c * P[m]["test"] - Y["test"]) ** 2).mean())
                entry[m]["test_mse_cal_from_val"] = round(cal, 6)
                entry[m]["test_gain_pct"] = round((raw - cal) / raw * 100, 1)
            ds_entry["seeds"][str(seed)] = entry
            print(f"[{ds} seed{seed}] y_std={ds_entry['y_std']}  "
                  f"vanilla c*(val)={entry['vanilla_lora']['val']['c_star']} "
                  f"test {entry['vanilla_lora']['test']['mse_raw']:.4f}->"
                  f"{entry['vanilla_lora']['test_mse_cal_from_val']:.4f} "
                  f"({entry['vanilla_lora']['test_gain_pct']:+.1f}%)  "
                  f"static {entry['static_ensemble']['test']['mse_raw']:.4f}->"
                  f"{entry['static_ensemble']['test_mse_cal_from_val']:.4f} "
                  f"({entry['static_ensemble']['test_gain_pct']:+.1f}%)", flush=True)
        res[ds] = ds_entry

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "calib_protocol.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("[saved]", p, flush=True)


if __name__ == "__main__":
    main()
