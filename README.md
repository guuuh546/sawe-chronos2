# SAWE on Chronos-2

Code and per-seed results for the paper:

**Member Selection and Band-Wise Evaluation of Frequency-Domain LoRA on Chronos-2**  
Yu Huang, Qi Chen, Qiang Huang  
*Electronics* (MDPI), manuscript ID `electronics-4689638`

## Repository layout

- `code/` — implementation of the three fine-tuning methods, the spectrum-informed weighted ensemble (SAWE), member selection, band-wise evaluation, and calibration checks.
- `results/` — per-seed raw results and three-seed summaries used in the revision; `P0_results.json` contains the static-weight baseline, threshold sensitivity, and band-share results; `r1_01_results.md` indexes the consolidated per-seed files.
- `data/` — expected location for the public benchmark datasets (not included; see `data/README.md`).
- `requirements.txt` — Python dependencies.

## Environment

Python 3.10+ and PyTorch 2.0+ (CUDA is recommended for the full grid).

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -r requirements.txt
```

## Model checkpoint

The experiments use a backbone initialized from Chronos-2 encoder weights. The checkpoint is not redistributed in this repository.

Download `amazon/chronos-2` from Hugging Face and place `config.json` and `model.safetensors` under `loratime/chronos-2/`, or change the default path in `code/src/foundation_models.py` (and `code/src/models.py`). See Section 3.5 of the paper for the exact backbone configuration.

## Data

The public benchmark datasets are not included. Place them as described in [`data/README.md`](data/README.md), or adjust the `data_dir` defaults in `code/src/datasets.py`.

Datasets used in the paper: ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange Rate, and Weather (`seq_len=48`, `pred_len=24`).

## Reproducing the main results

Main table (single methods and equal-weight ensemble):

```bash
python code/run_ensemble_experiments.py --e1 \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate weather \
  --seeds 42 123 456 --output_dir outputs/e1
```

Ablations (router input and member set):

```bash
python code/run_ensemble_experiments.py --e2 \
  --datasets exchange_rate ETTh2 --seeds 42 123 456 --output_dir outputs/e2
```

Conditional early exit:

```bash
python code/run_ensemble_experiments.py --e3 \
  --datasets exchange_rate ETTh2 --seeds 42 123 456 --output_dir outputs/e3
```

Cross-dataset transfer:

```bash
python code/run_ensemble_experiments.py --e4 \
  --datasets ETTh1 ETTh2 --seeds 42 --output_dir outputs/e4
```

Interpretability / weight statistics:

```bash
python code/run_ensemble_experiments.py --e5 \
  --datasets exchange_rate ETTh2 --seeds 42 --output_dir outputs/e5
```

Multi-seed matrix and validation-only member selection:

```bash
python code/_multiseed.py --datasets ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate weather --seeds 42 123 456
python code/_full_pipeline.py --stage val --datasets ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate weather --seeds 42 123 456
```

Band-wise and calibration scripts:

```bash
python code/_band_divided.py
python code/_band_report.py
python code/_calib_protocol.py
```

Run any script with `--help` for the full argument list.

## Included results

`results/` contains:

- `multiseed_<dataset>/summary.json` — three-seed summary for each dataset.
- `multiseed_<dataset>/raw_*.json` — per-seed raw results.
- `P0_results.json` — static-weight baseline, threshold sensitivity, and band shares; `P0_results.md` is a short note on its scope.
- `r1_01_results.md` — index of the per-seed result files.

## Notes

- `SpecGate-LoRA` is the name used in the paper for the spectrum-gated low-rank adapter; the internal code/configuration key is `freq_lora`.
- Model checkpoints (`.pt`) and per-sample predictions (`.npz`) are not included because of size; they can be regenerated with the provided scripts.
- The marked-up manuscript used during peer review is not part of this repository.

## Contact

Qiang Huang, `13499@sicau.edu.cn`
