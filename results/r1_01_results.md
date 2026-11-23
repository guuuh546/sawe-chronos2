# Consolidated results

The authoritative per-seed and three-seed results are stored in
`results/multiseed_<dataset>/`:

- `summary.json` — three-seed mean/std, per-band values, and band-edge definitions.
- `raw_<dataset>_seed<seed>.json` — raw per-seed results.

The manuscript tables are generated from these files. For the static-weight
baseline, threshold sensitivity, and band shares, see `P0_results.json`.
