"""
多种子重跑（4.4 消融 + 4.5 分带 + 4.6 路由诊断）
=================================================
目的：把原先的"诊断性单种子实验"升级为 3 种子，并**保存逐种子数值**。

阶段（--stage）：
  band      分带评估（4.5）：各成员与静态集成的分带 MSE + 误差能量占比
  ablation  消融（4.4）：特征组合 4 组 + 成员组合 4 组（仅 ABL_DATASETS）
  diag      路由诊断（4.6）：Oracle 收益上限、多数类基线、时间置换对照、频带 T1 领先
  all       以上全部（按数据集自动取舍）

设计：
  - 成员每个 (数据集, 种子) 只训练一次，供所有阶段复用
  - 结果落盘为 raw_{ds}_seed{s}.json（按阶段增量），支持断点续跑
  - 每次运行结束重新聚合出 summary.json（含 mean/std 与逐种子值）

用法：
    python -u _multiseed.py --stage all --datasets ETTm2
    python -u _multiseed.py --stage diag --datasets exchange_rate ETTh2
    python -u _multiseed.py --summary-only
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_CACHE_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_ensemble_experiments as R
from _band_divided import freq_contribution
from _band_report import make_bands
from src.ensemble import build_router_cache, train_router_cached
from src.datasets import get_dataloaders_for_experiment

SEEDS = [42, 123, 456]
BAND_DATASETS = ["exchange_rate", "ETTh2", "ETTm2"]
ABL_DATASETS = ["exchange_rate", "ETTh2"]
METHODS = ["vanilla_lora", "fourier_ft", "freq_lora"]
# 输出目录：默认沿用旧的 outputs/multiseed（复现旧结果用）；
# 改稿重跑必须显式 --out outputs/revision/R1-01/multiseed，避免覆盖旧结果。
OUT = os.environ.get("REVISION_OUT", os.path.join("outputs", "multiseed"))
LAG = R.PRED_LEN  # 因果滞后：窗口 i 可用的最新误差来自 i - pred_len

FEATURE_CONFIGS = {
    "PSD_only": {"use_csd": False, "use_stats": False},
    "PSD+CSD": {"use_csd": True, "use_stats": False},
    "PSD+Stats": {"use_csd": False, "use_stats": True},
    "Full": {"use_csd": True, "use_stats": True},
}


@torch.no_grad()
def predict(model, dataset, batch=64):
    model.eval()
    return torch.cat([model(x.to(R.DEVICE)) for x, _ in DataLoader(
        dataset, batch_size=batch, shuffle=False)], 0).cpu()


def collect_y(dataset, batch=256):
    return torch.cat([y for _, y in DataLoader(dataset, batch_size=batch, shuffle=False)], 0)


def band_mse(pred, y, masks):
    c = freq_contribution(pred - y)
    return [float(c[m.bool()].sum()) for m in masks]


def build_injected(method, input_dim, seed):
    """按统一管线构建注入后的模型（不训练），用于从缓存恢复成员权重"""
    R.set_seed(seed)
    model = R.build_base_model(input_dim).to(R.DEVICE)
    tm = R.get_target_modules(R.BACKBONE)
    if method == "vanilla_lora":
        model = R.inject_lora_to_model(model, rank=16, alpha=1.0, use_gating=False,
                                       use_spectral_reg=False, target_modules=tm)
    elif method == "freq_lora":
        model = R.inject_lora_to_model(model, rank=16, alpha=1.0, use_gating=True,
                                       use_spectral_reg=True, num_freq_bins=R.NUM_FREQ_BINS,
                                       lambda_l1=0.0001, target_modules=tm)
    elif method == "fourier_ft":
        # 与 run_ensemble_experiments.train_single_model 保持一致：忠实 FourierFT
        model = R.inject_fourier_ft_faithful(
            model, n_frequency=1000, scaling=150.0, random_loc_seed=777,
            target_modules=tm)
    elif method == "dense_spectral":
        # 附录消融：全频带稠密频谱 ΔW（旧实现，不是 FourierFT）
        model = R.inject_dense_spectral_dw(
            model, keep_ratio=0.5, target_modules=tm)
    else:
        raise ValueError(method)
    # 注意：注入的 LoRA 子模块是在 CPU 上创建的，必须在注入后统一移到设备
    return model.to(R.DEVICE)


def mem_path(ds, seed):
    return os.path.join(OUT, f"mem_{ds}_seed{seed}.pt")


def preds_path(ds, seed):
    return os.path.join(OUT, f"preds_{ds}_seed{seed}.npz")


def preds_dir(ds, seed):
    """D4/R1-06/R1-09/R1-10 复算用的逐 (数据集, 种子) 预测目录。"""
    return os.path.join(OUT, "predictions", f"{ds}_seed{seed}")


def save_member_preds(ds, seed, models, dls, preds_test):
    """保存逐成员 val/test 预测，供静态权重基线、分带与校准复算复用。

    落两份（内容等价，避免下游各自解析）：
      - ``preds_{ds}_seed{seed}.npz``：单文件，键 ``y_val`` / ``y_test`` /
        ``{val,test}_{method}``；
      - ``predictions/{ds}_seed{seed}/preds_{val,test}.npz``：目录式，
        另含等权集成 ``{val,test}_static_ensemble``。
    数组形状均为 (N, T, C)。
    """
    arrays = {
        "y_val": collect_y(dls["val_dataset"]).numpy(),
        "y_test": collect_y(dls["test_dataset"]).numpy(),
    }
    for m in METHODS:
        arrays[f"val_{m}"] = predict(models[m], dls["val_dataset"]).numpy()
        arrays[f"test_{m}"] = preds_test[m].numpy()
    arrays["val_static_ensemble"] = np.mean(
        [arrays[f"val_{m}"] for m in METHODS], axis=0)
    arrays["test_static_ensemble"] = np.mean(
        [arrays[f"test_{m}"] for m in METHODS], axis=0)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(preds_path(ds, seed), **arrays)

    d = preds_dir(ds, seed)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "preds_val.npz"),
                        y=arrays["y_val"],
                        **{m: arrays[f"val_{m}"] for m in METHODS},
                        static_ensemble=arrays["val_static_ensemble"])
    np.savez_compressed(os.path.join(d, "preds_test.npz"),
                        y=arrays["y_test"],
                        **{m: arrays[f"test_{m}"] for m in METHODS},
                        static_ensemble=arrays["test_static_ensemble"])
    print(f"    [preds] 已保存 {preds_path(ds, seed)} 与 {d}/preds_{{val,test}}.npz",
          flush=True)


def save_router_weights(ds, seed, W_val, W_test):
    """保存 SAWE 路由器在 val/test 上的逐样本权重 (N, 3)。"""
    d = preds_dir(ds, seed)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "router_weights_val.npz"),
                        weights=W_val, methods=np.array(METHODS))
    np.savez_compressed(os.path.join(d, "router_weights_test.npz"),
                        weights=W_test, methods=np.array(METHODS))
    print(f"    [router] 已保存 {d}/router_weights_{{val,test}}.npz", flush=True)


def train_or_load_members(ds, seed, dls, input_dim):
    """训练 3 个成员；若已有权重缓存则直接载入（省去重复训练）"""
    p = mem_path(ds, seed)
    models = {}
    if os.path.exists(p):
        print(f"    [cache] 载入成员权重 {p}", flush=True)
        ckpt = torch.load(p, map_location=R.DEVICE)
        for m in METHODS:
            model = build_injected(m, input_dim, seed)
            sd = model.state_dict()
            sd.update({k: v.to(R.DEVICE) for k, v in ckpt[m].items()})
            model.load_state_dict(sd)
            models[m] = model
        return models

    R.set_seed(seed)
    for m in METHODS:
        print(f"    [train] {m} (seed={seed})", flush=True)
        model, bvm = R.train_single_model(
            m, None, dls["train"], dls["val"], input_dim, R.DEVICE, seed)
        print(f"      best_val_mse={bvm:.4f}", flush=True)
        models[m] = model

    os.makedirs(OUT, exist_ok=True)
    ckpt = {m: {n: pm.detach().cpu() for n, pm in models[m].named_parameters()
                if pm.requires_grad} for m in METHODS}
    torch.save(ckpt, p)
    print(f"    [cache] 已保存成员权重 {p}", flush=True)
    return models


def compute_diag(preds, Y_te, K):
    """4.6 路由诊断：全部为推理计算，不需要额外训练。

    preds: {method: (N, T, C)} 测试集预测；Y_te: (N, T, C)
    """
    errs = {m: ((preds[m] - Y_te) ** 2).mean(dim=(1, 2)) for m in METHODS}  # 逐样本 MSE
    stack = torch.stack([errs[m] for m in METHODS], 0)                      # (K, N)

    best_idx = stack.argmin(0)                       # 每个样本的最优成员
    per_sample_best = stack.min(0).values
    oracle = float(per_sample_best.mean())

    fixed = {m: float(errs[m].mean()) for m in METHODS}   # 各成员的固定选择 MSE
    best_fixed_name = min(fixed, key=fixed.get)
    best_fixed = fixed[best_fixed_name]
    best_fixed_idx = METHODS.index(best_fixed_name)

    # 收益上限：逐样本 oracle 相对最优固定成员的提升
    oracle_gain_pct = (best_fixed - oracle) / best_fixed * 100

    # 多数类基线：始终选择"在样本中出现最多的最优成员"
    counts = torch.bincount(best_idx, minlength=len(METHODS)).float() / len(best_idx)
    maj_idx = int(counts.argmax())
    majority_acc = float(counts[maj_idx])                 # 与逐样本最优的一致率
    majority_mse = float(errs[METHODS[maj_idx]].mean())

    # 时间置换对照（严格因果）：窗口 i 只能用 i-LAG 处的最优成员
    n = len(best_idx)
    if n > LAG:
        lagged = torch.cat([best_idx[:1].expand(LAG), best_idx[:-LAG]])
        lag_acc = float((lagged == best_idx).float().mean())
        lag_mse = float(stack[lagged, torch.arange(n)].mean())
    else:
        lag_acc, lag_mse = float('nan'), float('nan')

    # 时间置换参考：把滞后序列整体环形错位 n//2，破坏时间对应关系
    if n > LAG:
        shift = n // 2
        permuted = torch.roll(lagged, shift)
        perm_mse = float(stack[permuted, torch.arange(n)].mean())
    else:
        perm_mse = float('nan')

    return {
        "per_member_fixed_mse": {m: round(fixed[m], 6) for m in METHODS},
        "best_fixed_member": best_fixed_name,
        "best_fixed_mse": round(best_fixed, 6),
        "oracle_mse": round(oracle, 6),
        "oracle_gain_pct": round(oracle_gain_pct, 3),
        "best_member_share": [round(float(c), 4) for c in counts],
        "majority_member": METHODS[maj_idx],
        "majority_acc": round(majority_acc, 4),
        "majority_mse": round(majority_mse, 6),
        "lag_acc": round(lag_acc, 4),
        "lag_mse": round(lag_mse, 6),
        "lag_gain_pct": round((best_fixed - lag_mse) / best_fixed * 100, 3),
        "time_permuted_mse": round(perm_mse, 6),
        "time_permuted_gain_pct": round((best_fixed - perm_mse) / best_fixed * 100, 3),
        "best_fixed_idx": best_fixed_idx,
    }


def build_caches(models, dls, router=None, keep_x=True):
    """为路由器训练构建缓存（成员预测 + raw 频谱特征），每个 (数据集,种子) 只建一次。

    与 `R.train_router` 数值等价：成员冻结 + eval，按索引顺序前向；构建过程不消费
    全局 RNG（并显式保存/还原 RNG 状态），因此不改变后续 router 初始化与置乱序列
    （见 revision_work/R1-01/test_router_cache_equiv.py 与
    test_router_feature_cache.py）。

    Args:
        router: 传入时同时缓存 `extract_raw(x)`（Full 特征的 raw 维度），
            避免 router 每个 epoch 重算 PSD + K 次 eigvalsh
        keep_x: 是否保留原始输入（消融需要为不同特征配置重算 raw_feat 时必须保留）
    """
    members = [models[m] for m in METHODS]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    t0 = time.time()
    try:
        ctr = build_router_cache(members, dls["train"], R.DEVICE,
                                 router=router, keep_x=keep_x)
        cva = build_router_cache(members, dls["val"], R.DEVICE,
                                 router=router, keep_x=keep_x)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    mem = ctr.memory_bytes()
    print(f"    [cache] members+raw train N={ctr.n_samples} / val N={cva.n_samples} "
          f"({time.time() - t0:.1f}s)；raw_feat "
          f"{'-' if ctr.raw_feat is None else tuple(ctr.raw_feat.shape)}，"
          f"内存 train={mem['total'] / 2 ** 20:.0f} MiB"
          f"（P={mem['P'] / 2 ** 20:.0f} / raw={mem['raw_feat'] / 2 ** 20:.0f} / "
          f"X={mem['X'] / 2 ** 20:.0f}）", flush=True)
    return ctr, cva


def caches_with_features(base, extractor):
    """为某个特征配置构建 raw_feat（共享成员预测，只重算特征）。"""
    if base[0].raw_feat is not None and base[0].raw_feat.shape[1] == _raw_dim(extractor):
        return base
    return (base[0].with_raw(extractor, R.DEVICE),
            base[1].with_raw(extractor, R.DEVICE))


def _raw_dim(ext):
    d = ext.K * ext.n_features
    if ext.use_csd:
        d += ext.K * 4
    if ext.use_stats:
        d += ext.n_features * 4
    return d


def train_router_fast(ens, caches):
    """缓存版路由器训练（与 R.train_router 数值等价）。"""
    t0 = time.time()
    hist = train_router_cached(ens, caches[0], caches[1], lr=1e-3,
                               num_epochs=20, device=R.DEVICE)
    print(f"    [router] 缓存训练完成 {time.time() - t0:.1f}s "
          f"({len(hist['val_loss'])} epochs)", flush=True)
    return hist


def run_one(ds, seed, stage, have, need_preds=False):
    """跑一个 (数据集, 种子) 的指定阶段；have 为该文件已有内容的副本"""
    need_band = stage in ("band", "all") and "band" not in have
    need_abl = stage in ("ablation", "all") and ds in ABL_DATASETS and "ablation" not in have
    need_diag = stage in ("diag", "all") and "diag" not in have
    need_extra = stage in ("extra", "all") and "extra" not in have
    if not (need_band or need_abl or need_diag or need_extra or need_preds):
        return None

    dls = get_dataloaders_for_experiment(
        ds, batch_size=R.BATCH_SIZE, seq_len=R.SEQ_LEN, pred_len=R.PRED_LEN)
    x0, _ = next(iter(dls["train"]))
    input_dim = x0.shape[-1]
    Y_te = collect_y(dls["test_dataset"])
    K = R.PRED_LEN // 2 + 1
    masks, _, edges = make_bands(K)

    models = train_or_load_members(ds, seed, dls, input_dim)
    out = dict(have)
    out.update({"dataset": ds, "seed": seed, "input_dim": input_dim})

    with torch.no_grad():
        preds = {m: predict(models[m], dls["test_dataset"]) for m in METHODS}

    if need_preds:
        save_member_preds(ds, seed, models, dls, preds)

    # 路由器（extra / ablation）统一走缓存路径：只建一次，各路由器复用。
    # extra 阶段会创建主路由器（Full 特征），这里提前创建以便一并缓存 raw 频谱特征，
    # 使 router 训练不再每个 epoch 重算 PSD + K 次 eigvalsh。
    # 注意：只在 need_extra 时提前创建，避免在"只跑消融"时多消耗一次 RNG、
    # 改变消融路由器的初始化（那样就与旧结果不可比了）。
    main_router = None
    caches = None
    if need_extra:
        main_router = R.SpectrumRouter(
            n_features=input_dim, seq_len=R.SEQ_LEN, num_models=len(METHODS),
            hidden_dim=128, use_csd=True, use_stats=True).to(R.DEVICE)
        caches = build_caches(models, dls, router=main_router, keep_x=need_abl)
    elif need_abl:
        caches = build_caches(models, dls, router=None, keep_x=True)

    # ---------- 分带评估 ----------
    if need_band:
        ens = torch.stack([preds[m] for m in METHODS], 0).mean(0)
        res = []
        with torch.no_grad():
            for i, (x, y) in enumerate(DataLoader(dls["train_dataset"], batch_size=64,
                                                  shuffle=False)):
                x, y = x.to(R.DEVICE), y.to(R.DEVICE)
                res.append((models["vanilla_lora"](x) - y).cpu())
                if i >= 19:
                    break
        ref = freq_contribution(torch.cat(res, 0))
        shares = [float(ref[m.bool()].sum() / ref.sum()) for m in masks]

        rows = {}
        for m in METHODS:
            rows[m] = {"overall": float(((preds[m] - Y_te) ** 2).mean()),
                       "per_band": band_mse(preds[m], Y_te, masks)}
        rows["static_ensemble"] = {"overall": float(((ens - Y_te) ** 2).mean()),
                                   "per_band": band_mse(ens, Y_te, masks)}
        out["band"] = {"edges": edges,
                       "error_energy_share": [round(s, 6) for s in shares],
                       "methods": {k: {"overall": round(v["overall"], 6),
                                       "per_band": [round(z, 6) for z in v["per_band"]]}
                                   for k, v in rows.items()}}
        print(f"    [band] overall={ {k: round(v['overall'], 4) for k, v in rows.items()} }",
              flush=True)

    # ---------- 路由诊断（4.6） ----------
    if need_diag:
        out["diag"] = compute_diag(preds, Y_te, K)
        d = out["diag"]
        print(f"    [diag] oracle_gain={d['oracle_gain_pct']}%  majority_acc={d['majority_acc']}  "
              f"lag_acc={d['lag_acc']}  lag_gain={d['lag_gain_pct']}%  "
              f"perm_gain={d['time_permuted_gain_pct']}%", flush=True)

    # ---------- 条件早退 + 校准检查（4.6 的另两项） ----------
    if need_extra:
        ex = {}
        # (a) 条件早退：路由器在测试集上的权重集中度与各阈值下的早退率
        # 复用上面提前创建的主路由器（参数与"在此处新建"逐位相同，见 work.md §13）
        router = main_router
        ens = R.SpectrumAwareEnsemble([models[m] for m in METHODS], router).to(R.DEVICE)
        train_router_fast(ens, caches)
        met = R.evaluate_model(ens, dls["test"], R.DEVICE)
        ws = []
        with torch.no_grad():
            for x, _ in DataLoader(dls["test_dataset"], batch_size=128, shuffle=False):
                ws.append(ens.router(x.to(R.DEVICE)).cpu())
        W = torch.cat(ws, 0)
        wv = []
        with torch.no_grad():
            for x, _ in DataLoader(dls["val_dataset"], batch_size=128, shuffle=False):
                wv.append(ens.router(x.to(R.DEVICE)).cpu())
        W_val = torch.cat(wv, 0)
        save_router_weights(ds, seed, W_val.numpy(), W.numpy())
        maxw = W.max(dim=1).values
        ths = [0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
        ex["early_exit"] = {
            "sawe_mse": round(float(met["mse"]), 6),
            "num_samples": int(W.shape[0]),
            "max_weight_mean": round(float(maxw.mean()), 4),
            "max_weight_p50": round(float(maxw.median()), 4),
            "max_weight_p90": round(float(maxw.quantile(0.9)), 4),
            "max_weight_max": round(float(maxw.max()), 4),
            "weight_mean_per_member": [round(float(z), 4) for z in W.mean(0)],
            "exit_rate": {str(t): round(float((maxw >= t).float().mean()), 4) for t in ths},
        }
        del ens, router
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # (b) 校准检查：在验证集上拟合标量/线性组合，观察其在测试集的表现
        with torch.no_grad():
            pv = {m: predict(models[m], dls["val_dataset"]).reshape(-1) for m in METHODS}
            pt = {m: preds[m].reshape(-1) for m in METHODS}
            yv = collect_y(dls["val_dataset"]).reshape(-1)
            yt = Y_te.reshape(-1)
        Pv = torch.stack([pv[m] for m in METHODS], 1).double()
        Pt = torch.stack([pt[m] for m in METHODS], 1).double()
        yvd, ytd = yv.double(), yt.double()
        best_fixed = min(float(((pt[m] - yt) ** 2).mean()) for m in METHODS)
        mu_v, mu_t = Pv.mean(1), Pt.mean(1)
        c = float((mu_v @ yvd) / (mu_v @ mu_v))          # 标量缩放系数
        alpha = torch.linalg.lstsq(Pv, yvd.unsqueeze(1)).solution.squeeze(1)
        ex["calib"] = {
            "best_fixed_test_mse": round(best_fixed, 6),
            "scalar_c": round(c, 4),
            "scalar_val_mse": round(float(((c * mu_v - yvd) ** 2).mean()), 6),
            "scalar_test_mse": round(float(((c * mu_t - ytd) ** 2).mean()), 6),
            "linear_alpha": [round(float(a), 3) for a in alpha],
            "linear_val_mse": round(float(((Pv @ alpha - yvd) ** 2).mean()), 6),
            "linear_test_mse": round(float(((Pt @ alpha - ytd) ** 2).mean()), 6),
        }
        out["extra"] = ex
        ee, cb = ex["early_exit"], ex["calib"]
        print(f"    [early_exit] maxw p50={ee['max_weight_p50']} p90={ee['max_weight_p90']} "
              f"max={ee['max_weight_max']} exit@0.40={ee['exit_rate']['0.4']} "
              f"exit@0.35={ee['exit_rate']['0.35']}", flush=True)
        print(f"    [calib] alpha={cb['linear_alpha']} val={cb['linear_val_mse']} "
              f"test={cb['linear_test_mse']} (最优单模 {cb['best_fixed_test_mse']})", flush=True)

    # ---------- 消融 ----------
    if need_abl:
        abl = {}
        for cfg_name, cfg in FEATURE_CONFIGS.items():
            router = R.SpectrumRouter(n_features=input_dim, seq_len=R.SEQ_LEN,
                                      num_models=len(METHODS), hidden_dim=128, **cfg).to(R.DEVICE)
            ens = R.SpectrumAwareEnsemble([models[m] for m in METHODS], router).to(R.DEVICE)
            # 特征配置不同 ⇒ raw_feat 维度不同，各自建一次（成员预测复用）
            cfg_caches = caches_with_features(caches, router.spectral_extractor)
            train_router_fast(ens, cfg_caches)
            met = R.evaluate_model(ens, dls["test"], R.DEVICE)
            abl[f"feature_{cfg_name}"] = {"mse": float(met["mse"]), "mae": float(met["mae"])}
            print(f"    [ablation][feature] {cfg_name}: MSE={met['mse']:.4f}", flush=True)
            del ens, router, cfg_caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        combos = {
            "All_3": METHODS,
            "No_FourierFT": [m for m in METHODS if m != "fourier_ft"],
            "No_FreqLoRA": [m for m in METHODS if m != "freq_lora"],
            "No_VanillaLoRA": [m for m in METHODS if m != "vanilla_lora"],
        }
        for cfg_name, combo in combos.items():
            router = R.SpectrumRouter(n_features=input_dim, seq_len=R.SEQ_LEN,
                                      num_models=len(combo), hidden_dim=128,
                                      use_csd=True, use_stats=True).to(R.DEVICE)
            ens = R.SpectrumAwareEnsemble([models[m] for m in combo], router).to(R.DEVICE)
            # 成员子集复用同一份缓存（仅切出对应成员通道）
            if combo == METHODS:
                sub = caches
            else:
                idxs = [METHODS.index(m) for m in combo]
                sub = (caches[0].subset(idxs), caches[1].subset(idxs))
            train_router_fast(ens, sub)
            met = R.evaluate_model(ens, dls["test"], R.DEVICE)
            abl[f"combo_{cfg_name}"] = {"mse": float(met["mse"]), "mae": float(met["mae"])}
            print(f"    [ablation][combo] {cfg_name}: MSE={met['mse']:.4f}", flush=True)
            del ens, router
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out["ablation"] = abl

    for m in list(models.keys()):
        del models[m]
    caches = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def save_raw(res):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"raw_{res['dataset']}_seed{res['seed']}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"  [saved] {p}", flush=True)


def ms(vals):
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "per_seed": [float(v) for v in a]}


def aggregate():
    files = [f for f in os.listdir(OUT) if f.startswith("raw_")] if os.path.isdir(OUT) else []
    raws = {}
    for f in files:
        with open(os.path.join(OUT, f), encoding="utf-8") as fh:
            r = json.load(fh)
        raws[f"{r['dataset']}|{r['seed']}"] = r

    known = []
    for key in raws:
        ds = key.split("|")[0]
        if ds not in known:
            known.append(ds)

    summary = {}
    for ds in known:
        seeds = [s for s in SEEDS if f"{ds}|{s}" in raws]
        if not seeds:
            continue
        entry = {"seeds": seeds, "n_seeds": len(seeds)}

        band_seeds = [raws[f"{ds}|{s}"]["band"] for s in seeds if "band" in raws[f"{ds}|{s}"]]
        if band_seeds:
            entry["band_edges"] = band_seeds[0]["edges"]
            nb = len(band_seeds[0]["error_energy_share"])
            entry["error_energy_share"] = [ms([b["error_energy_share"][j] for b in band_seeds])
                                           for j in range(nb)]
            methods = {}
            for k in band_seeds[0]["methods"]:
                nbb = len(band_seeds[0]["methods"][k]["per_band"])
                methods[k] = {
                    "overall": ms([b["methods"][k]["overall"] for b in band_seeds]),
                    "per_band": [ms([b["methods"][k]["per_band"][j] for b in band_seeds])
                                 for j in range(nbb)],
                }
            entry["methods"] = methods

        abl_seeds = [raws[f"{ds}|{s}"]["ablation"] for s in seeds if "ablation" in raws[f"{ds}|{s}"]]
        if abl_seeds:
            entry["ablation"] = {k: {"mse": ms([a[k]["mse"] for a in abl_seeds]),
                                     "mae": ms([a[k]["mae"] for a in abl_seeds])}
                                 for k in abl_seeds[0].keys()}

        diag_seeds = [raws[f"{ds}|{s}"]["diag"] for s in seeds if "diag" in raws[f"{ds}|{s}"]]
        if diag_seeds:
            keys = [k for k, v in diag_seeds[0].items()
                    if isinstance(v, (int, float)) and k not in
                    ("best_fixed_idx",)]
            d = {k: ms([g[k] for g in diag_seeds]) for k in keys}
            d["per_member_fixed_mse"] = {m: ms([g["per_member_fixed_mse"][m] for g in diag_seeds])
                                         for m in METHODS}
            d["best_fixed_member"] = [g["best_fixed_member"] for g in diag_seeds]
            d["majority_member"] = [g["majority_member"] for g in diag_seeds]
            d["best_member_share"] = [g["best_member_share"] for g in diag_seeds]
            entry["diag"] = d

        extra_seeds = [raws[f"{ds}|{s}"]["extra"] for s in seeds if "extra" in raws[f"{ds}|{s}"]]
        if extra_seeds:
            ee0, cb0 = extra_seeds[0]["early_exit"], extra_seeds[0]["calib"]
            ee = {}
            for k in ee0:
                if isinstance(ee0[k], (int, float)):
                    ee[k] = ms([g["early_exit"][k] for g in extra_seeds])
                elif isinstance(ee0[k], list):
                    ee[k] = [ms([g["early_exit"][k][j] for g in extra_seeds])
                             for j in range(len(ee0[k]))]
                elif isinstance(ee0[k], dict):
                    ee[k] = {t: ms([g["early_exit"][k][t] for g in extra_seeds])
                             for t in ee0[k]}
            cb = {}
            for k in cb0:
                if isinstance(cb0[k], list):
                    cb[k] = [ms([g["calib"][k][j] for g in extra_seeds])
                             for j in range(len(cb0[k]))]
                elif isinstance(cb0[k], (int, float)):
                    cb[k] = ms([g["calib"][k] for g in extra_seeds])
            entry["extra"] = {"early_exit": ee, "calib": cb}

        summary[ds] = entry

    p = os.path.join(OUT, "summary.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] {p}", flush=True)
    return summary


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["band", "ablation", "diag", "extra", "all"])
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 outputs/multiseed；改稿重跑应显式指定独立目录）")
    args = ap.parse_args()

    if args.out:
        OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    print(f"[out] {OUT}", flush=True)
    if args.summary_only:
        aggregate()
        return

    seeds = args.seeds or SEEDS
    datasets = args.datasets or BAND_DATASETS

    total = len(datasets) * len(seeds)
    done = 0
    for ds in datasets:
        for seed in seeds:
            done += 1
            p = os.path.join(OUT, f"raw_{ds}_seed{seed}.json")
            have = {}
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    have = json.load(f)
                need_band = args.stage in ("band", "all") and "band" not in have
                need_abl = (args.stage in ("ablation", "all") and ds in ABL_DATASETS
                            and "ablation" not in have)
                need_diag = args.stage in ("diag", "all") and "diag" not in have
                need_extra = args.stage in ("extra", "all") and "extra" not in have
                need_preds = not os.path.exists(preds_path(ds, seed))
                if not (need_band or need_abl or need_diag or need_extra or need_preds):
                    print(f"[skip] {ds} seed={seed} 该阶段已完成 "
                          f"({done}/{total})", flush=True)
                    continue
            else:
                need_preds = True
            print(f"\n{'='*60}\n[progress {done}/{total}] {ds} seed={seed} "
                  f"stage={args.stage}\n{'='*60}", flush=True)
            res = run_one(ds, seed, args.stage, have, need_preds=need_preds)
            if res is not None:
                save_raw(res)
            print(f"[progress {done}/{total}] {ds} seed={seed} 完成", flush=True)

    aggregate()


if __name__ == "__main__":
    main()
