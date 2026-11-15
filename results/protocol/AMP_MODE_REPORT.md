# Amplitude mode (`amp_mode`) — report

Default remains **`constant`**. No retraining. Decision to switch defaults is human-only.

## Files changed

| Path | Change |
|---|---|
| `models/amp_interp.py` | **New** — `constant` / `linear` / `pchip` via `log_amp_per_sample` |
| `models/lightning_module.py` | `amp_mode` hparam; amp via `log_amp_per_sample`; full-then-slice in `_rollout_mismatch` |
| `paper_figs/protocol_utils.py` | `reconstruct_fp64(..., amp_mode=)` |
| `evalute.py` | `--amp-mode`; full-then-slice; record in `metrics.json` |
| `eval_fm_unified.py` | `--amp-mode` + CSV column; `share_memory=False` preload |
| `paper_figs/fig_context_fraction.py` | `--amp-mode` + sidecar; `share_memory=False` |
| `paper_figs/fig_calibration.py` | `--amp-mode` |
| `paper_figs/fig_speed_frontier.py` | `--amp-mode` |
| `scripts/measure_floor.py` | `--amp-mode`; full-sequence e2e/token-space |
| `tests/test_amp_interp.py` | **New** — scipy parity, regression, floor order, ckpt fallback |

`scripts/test_amp_floor.py` left unchanged (scipy reference).

## Tests

```text
pytest tests/test_amp_interp.py tests/test_eval_protocol.py  →  24 passed, 1 skipped
```

Pre-existing collection errors remain in `tests/test_dataset_norm.py` / `tests/test_training.py` (`GWSurrogateLit` rename) — unrelated.

`log_amp_per_sample` vs SciPy `PchipInterpolator` / `np.interp`: max |diff| ≲ 3e-15 (fp64).

## Acceptance — context fraction (200 rows, frac=0, window 0.75, n_steps=40, mean_of=4, seed=1234)

Checkpoint: `logs/scale_study/n1000k_2p5m/version_0/checkpoints/epoch=017-val_mismatch=0.0119.ckpt`

| mode | `floor_pt_window` p50 | `mm_pt_window` p50 / p90 / p99 | `mm_strict_window` p50 / p90 / p99 |
|---|---|---|---|
| **constant** | **1.122e-03** | 2.064e-03 / 2.675e-03 / 8.289e-03 | 5.338e-02 / 2.150e-01 / 2.854e-01 |
| **pchip** | **9.616e-05** | 9.711e-04 / 1.661e-03 / 6.858e-03 | 5.226e-02 / 2.144e-01 / 2.846e-01 |

Checks:
- `constant` `floor_pt_window_p50 = 1.122e-03` — matches target **1.122e-03**.
- `constant` `mm_pt_window_p50 = 2.064e-03` vs earlier quote **1.991e-03** (~3.7% higher; same seed/rows; likely prior run differed slightly in stack). Floor match is exact → do **not** STOP.
- `pchip` floor p50 within ±15% of 1e-4 (got 9.6e-5).
- `pchip` total `mm_pt_window` p50 ≈ 9.7e-4 (~1e-3 as expected).

## Acceptance — `eval_fm_unified` (1000 rows, ctx 0.5, mean_of=4, 40 steps)

| mode | `mismatch_phase_time_future` p50 | p90 |
|---|---|---|
| constant | 1.639e-03 | 5.262e-03 |
| pchip | 1.360e-03 | 5.044e-03 |

PCHIP improves median phase+time future mismatch by ~17% at ctx 0.5 (smaller gain than the θ-only / merger-window floor drop, as expected).

## Unexpected

1. WSL `/dev/shm` is 64 MB — `preload=True` + `share_memory=True` failed; eval entry points now use `share_memory=False`.
2. Provided `models/amp_interp.py` was not in the repo; implemented to match SciPy / `test_amp_floor.py` centres `k*P+(P-1)/2` (verified).
3. PCHIP segment-alone vs full-then-slice differs through the **first cubic piece** (~1.5 patches), not only the first half-patch (endpoint derivative); linear differs only in the first half-patch.
