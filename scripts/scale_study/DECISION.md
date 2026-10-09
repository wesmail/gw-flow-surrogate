# Decision checklist (plan step 6)

Fill this in **after** you have:

1. `results/scale_study/prediction.json` (written *before* looking at 300k)
2. Eval CSV for `300k_2p5m`
3. Eval CSVs for `120k_0p6m` and `120k_10m`

## Headline numbers

| Quantity | Value |
|----------|-------|
| Predicted median @ 300k (no floor) | |
| Predicted N for 10⁻⁴ (no floor) | |
| Predicted median @ 300k (with floor) | |
| Predicted N for 10⁻⁴ (with floor) | |
| Measured median @ 300k | |
| Measured median @ 120k / 0.6M | |
| Measured median @ 120k / 2.5M | |
| Measured median @ 120k / 10M | |

## Decision table

| 300k result | 10M model at 120k | Conclusion | Action |
|-------------|-------------------|------------|--------|
| Matches prediction | No better than 2.5M | Data is the only lever | Refit with all six points; take N★ for 10⁻⁴ |
| Above prediction | Clearly better | Model too small | Train 10M at 300k, then scale both |
| Above prediction | No better | Something else is the floor | Euler check (20/50/100) + optimisation before more data |

**Your branch:** _______________________

## Euler-step check (if needed)

```bash
bash scripts/scale_study/06_euler_check.sh 120k_2p5m
# and/or
bash scripts/scale_study/06_euler_check.sh 300k_2p5m
```

If 50 or 100 steps clearly beat 20, freeze the new `N_STEPS` in `PROTOCOL.env` and re-eval.

## Target run (plan step 7)

Only after the decision:

1. Generate additional waveforms if N★ > 300k (new pool or extend — document seed continuity).
2. Train at the chosen (N, model).
3. Report median / 95th percentile / maximum mismatch from the eval CSV.
4. Plot: `python scripts/scale_study/05_plot_scaling.py`
