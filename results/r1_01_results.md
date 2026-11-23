# R1-01 结果汇总（忠实 FourierFT 重跑）

自动生成：`revision_work/R1-01/merge_summaries.py`。所有数值为测试集 MSE 的 3 种子均值（std 见 summary.json）。

## 表 A. 成员与集成（新实现）

| 数据集 | vanilla | FourierFT(忠实) | Freq-LoRA | 等权集成 | SAWE | 最佳单方法 | SAWE≤最佳单模？ |
|---|---|---|---|---|---|---|---|
| exchange_rate | 0.4071 | 0.3921 | 0.3954 | 0.3781 | 0.3797 | 0.3921 | 是 |
| ETTh1 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | — |
| ETTh2 | 0.1624 | 0.1731 | 0.1706 | 0.1532 | 0.1531 | 0.1624 | 是 |
| ETTm1 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | — |
| ETTm2（partial） | 0.3603 | 0.1056 | 0.1153 | 0.1418 | 0.1064 | 0.1056 | 否 |
| Electricity | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | — |
| Weather | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | 待跑 | — |

- 已跑数据集：3/7；其中 SAWE ≤ 最佳单方法 的个数：**2/3**（原稿声称 7/7）。

## 表 B. Exchange Rate：旧实现 vs 忠实 FourierFT

| 方法 | 旧（dense spectral ΔW） | 新（忠实 FourierFT） | 变化 |
|---|---|---|---|
| vanilla_lora | 0.4071 | 0.4071 | +0.0% |
| FourierFT | 0.6245 | 0.3921 | -37.2% |
| Freq-LoRA | 0.3954 | 0.3954 | +0.0% |
| 等权集成 | 0.4446 | 0.3781 | -15.0% |

## 表 C. 消融：All_3 vs No_FourierFT（新实现）

| 数据集 | All_3 | No_FourierFT | 去掉 FourierFT 的增益 |
|---|---|---|---|
| exchange_rate | 0.3798 | 0.3824 | -0.7% |
| ETTh2 | 0.1531 | 0.1517 | +0.9% |

## 各数据集产物路径

- `exchange_rate`：`outputs\revision\R1-01\multiseed_exchange_rate`
- `ETTh1`：未完成
- `ETTh2`：`outputs\revision\R1-01\multiseed_ETTh2`
- `ETTm1`：未完成
- `ETTm2`：`outputs\revision\R1-01\multiseed_ETTm2`
- `Electricity`：未完成
- `Weather`：未完成
