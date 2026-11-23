# Data layout

The public benchmark datasets are **not** included in this repository. Place the CSV files under `data/` using the paths below, or change the `data_dir` defaults in `code/src/datasets.py`.

| Dataset | Expected path |
|---|---|
| ETTh1 | `data/ETDataset/ETTh1.csv` |
| ETTh2 | `data/ETDataset/ETTh2.csv` |
| ETTm1 | `data/ETDataset/ETTm1.csv` |
| ETTm2 | `data/ETDataset/ETTm2.csv` |
| Electricity | `data/electricity-20260606T074431Z-3-001/electricity/electricity.csv` |
| Exchange Rate | `data/exchange_rate-20260606T075627Z-3-001/exchange_rate/exchange_rate.csv` |
| Weather | `data/weather.csv` |

Each CSV should contain a date/time column followed by the numeric series used as input features. The loaders estimate normalization statistics from the training segment only and use a chronological 7:1:2 split.

The datasets are publicly available from their original sources (ETDataset, UCI Electricity, Exchange Rate, and Weather); see the references in the manuscript.
