# Task: add interpolated amplitude reconstruction (`amp_mode`) to gw-flow-surrogate

## Context (read first)

Tokens store, per patch of P = 32 samples, ONE number for amplitude: the mean of log_a over
samples [kP, kP+P-1]. `GWFlowSurrogateLit._reconstruct_waveform` turns tokens into h(t) and
currently draws that amplitude as a flat step over the patch. Phase is exact (per-sample residuals).

`scripts/test_amp_floor.py` measured the representation floor (TRUE tokens, no model, vs the
original waveform) on 1000 `data/n1000k` test waveforms:

| region | constant (today) | linear | pchip |
|---|---|---|---|
| full waveform, median | 4.92e-4 | 1.64e-4 | **4.47e-5** |
| last 50%, median | 7.47e-4 | 2.47e-4 | **6.71e-5** |
| last 25% (merger), median | 1.15e-3 | 3.73e-4 | **1.02e-4** |

PCHIP through the patch centres lowers the floor 11x with the same tokens and the same model.
Goal: make the amplitude rule selectable everywhere a waveform is reconstructed, verify that it
is exact and consistent, then re-evaluate an existing checkpoint with it. **No retraining.**

## Hard constraints

1. Do NOT change tokens, `datasets/patchify.py`, the training loss, the data pipeline, or any checkpoint.
2. Default behaviour must stay identical: `amp_mode="constant"` everywhere unless explicitly set.
   Old checkpoints (no `amp_mode` in hparams) must load and behave exactly as before.
3. One implementation only: every reconstruction path imports `models/amp_interp.py`.
   No second copy of the interpolation anywhere.
4. `models/lightning_module.py` has already been restored to the original patch-constant
   `_reconstruct_waveform` (the stray `"""..."""` block was removed). Only verify: there is exactly
   one `_reconstruct_waveform`, no leftover string block, no ad-hoc interpolation. Then apply Step 2.

## Step 1 — add `models/amp_interp.py` (provided, already verified)

Add the file `models/amp_interp.py` exactly as provided (it is attached next to this prompt). It exports
`AMP_MODES = ("constant", "linear", "pchip")` and
`log_amp_per_sample(la_patch (B, n), patch_len, mode) -> (B, n*patch_len)`.
It was verified against `scipy.interpolate.PchipInterpolator` (max |diff| 1.4e-14 in fp64, incl.
n = 2, 3, flat stretches) and reproduces `scripts/test_amp_floor.py` to 3e-14. Do not rewrite it.

## Step 2 — `models/lightning_module.py`

1. New constructor argument `amp_mode: str = "constant"` (after `use_abs_time`), validated against
   `AMP_MODES`; stored by `save_hyperparameters()` automatically.
2. In `_reconstruct_waveform`, replace ONLY the amplitude line
   `amp = torch.exp(la_phys)[:, :, None]` by
   ```python
   amp_mode = getattr(self.hparams, "amp_mode", "constant")
   la_t = log_amp_per_sample(la_phys, patch_len, amp_mode)           # (B, n*P)
   amp = torch.exp(la_t).view(B, n, patch_len)
   ```
   Phase code unchanged. Update the docstring.
3. **Edge rule (important):** interpolation needs neighbouring patches. A segment reconstructed on
   its own has no left neighbour, so its first half-patch is held flat. Therefore in
   `_rollout_mismatch`, reconstruct the FULL sequence (seed patches + generated patches, i.e.
   `cat(true[:ctx], gen)` for the prediction and `true[:ctx+n_gen]` for the truth) and then slice
   the generated samples `[ctx*P:]`. With `amp_mode="constant"` the result must be bitwise identical
   to the current code (slicing a flat-step reconstruction changes nothing).

## Step 3 — evaluation entry points

Add `--amp-mode {constant,linear,pchip}` (default `constant`) and record it in every output
(CSV column or sidecar JSON) in:

- `evalute.py` (`evaluate_batch` and CLI): set `lit.hparams.amp_mode = args.amp_mode` after loading,
  and apply the same full-sequence-then-slice rule to `h_true` / `h_pred` (and any teacher-forced
  reconstruction) as in Step 2.3.
- `eval_fm_unified.py`: pass through to `evaluate_batch` via `lit.hparams.amp_mode`; add an
  `amp_mode` column to the CSV.
- `paper_figs/fig_context_fraction.py`: its local fp64 `reconstruct()` must use
  `log_amp_per_sample(la, P, a.amp_mode)` (fp64) for the amplitude; it already reconstructs the full
  sequence. Add `amp_mode` to the `.json` sidecar. The floor columns must use the same `amp_mode`.
- `paper_figs/fig_calibration.py`, `paper_figs/fig_speed_frontier.py`, `scripts/measure_floor.py`,
  and `paper_figs/protocol_utils.py` (if it exists): same flag, same function.
- `scripts/test_amp_floor.py`: leave unchanged (it is the independent scipy reference).

## Step 4 — tests: `tests/test_amp_interp.py` (CPU, < 30 s)

1. `pchip` and `linear` equal scipy (`PchipInterpolator` / `np.interp` on centres
   `k*P + (P-1)/2`, clipped at the ends) to 1e-12 in fp64 for n in {2, 3, 5, 40, 320}, random and
   monotone data, and a flat stretch.
2. `constant` equals `repeat_interleave` exactly; n = 1 returns constant for every mode.
3. Regression: `_reconstruct_waveform` with `amp_mode="constant"` is bitwise equal to a copy of the
   original function (paste the original into the test as a reference).
4. Floor ordering on synthetic chirps (`scripts/make_smoke_cf.py` style data): patchify true tracks,
   reconstruct, mismatch vs original on the last 25%: pchip < linear < constant.
5. Old checkpoint compatibility: a checkpoint saved without `amp_mode` loads, and
   `hparams.amp_mode` falls back to `"constant"`.
6. Full-vs-segment: for `amp_mode="pchip"`, slicing a full reconstruction differs from
   reconstructing the segment alone only inside the first half-patch of the segment.

Run `pytest tests/ -v`; all existing tests must still pass.

## Step 5 — acceptance runs (GPU, report the numbers)

Use the checkpoint and data below; same seed, same rows for every run.

```bash
CKPT=logs/scale_study/n1000k_2p5m/version_0/checkpoints/epoch=017-val_mismatch=0.0119.ckpt
MAN=data/n1000k/manifest.parquet; NS=data/n1000k/norm_stats.npz
for AMP in constant pchip; do
python paper_figs/fig_context_fraction.py eval --checkpoint "$CKPT" --manifest $MAN \
  --norm-stats $NS --split test --n-waveforms 200 --fractions 0 --window-start 0.75 \
  --n-steps 40 --mean-of 4 --seed 1234 --amp-mode $AMP --out results/amp_$AMP.csv
done
```

Acceptance:
- `constant` reproduces the earlier run exactly: `mm_pt_window_p50 = 1.991e-03`,
  `floor_pt_window_p50 = 1.122e-03` (same seed, same rows). If not, STOP and report.
- `pchip`: `floor_pt_window_p50` close to 1.0e-4 (± 15%, it is 200 rows not 1000).
- Report `mm_pt_window` and `mm_strict_window` p50/p90/p99 for both modes side by side. Expected
  (not required): pchip total around 1e-3.
- Also run `eval_fm_unified.py` (ctx 0.5, 1000 rows, mean-of 4, 40 steps) with both modes on the same
  checkpoint and report `mismatch_phase_time_future` p50/p90.

## Deliverable

A short report (`results/protocol/AMP_MODE_REPORT.md`) with: files changed (diff summary), test
results, the acceptance table, and anything unexpected. Do NOT change any default to `pchip`;
that decision is made by the human after reading the report.
