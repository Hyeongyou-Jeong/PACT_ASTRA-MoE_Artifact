# Supplemental results (optional)

Reuse the nine Qwen3-30B-A3B traces. Primary configuration remains B=12, W=8,
F=4 under `ther/results/primary/`.

| File | Contents |
|---|---|
| `ther_wf_sensitivity.csv` | Cumulative Weighted-LFU / THER / Oracle over W∈{4,8,16,32}, F∈{2,4,8}, B=12 |
| `ther_wf_gaps.csv` | THER−Cumulative Weighted-LFU and Oracle−THER gaps |
| `ther_b_sensitivity.csv` | B∈{6,12,18,24} at W=8, F=4 |
| `mixed_ther_wf_sensitivity.{png,pdf}` | Mixed THER vs W for each F |
| `ther_minus_wlf_by_domain_W.{png,pdf}` | Domain gaps across W at F=4 |
| `matched_weighted_lfu_check.csv` | W-matched token-weighted Top-B vs THER (sanity; expect identical) |
| `sanity_flags.json` | Automatic anomaly flags |
| `sensitivity_meta.json` | Sweep metadata |

CSV policy key `WeightedLFU` denotes Cumulative Weighted-LFU (no finite W).

```bash
python3.10 ther/scripts/run_sensitivity.py --also-b-sweep
python3.10 ther/scripts/plot_supplemental.py
python3.10 ther/scripts/matched_weighted_lfu_check.py
```
