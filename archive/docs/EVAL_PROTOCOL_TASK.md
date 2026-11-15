# Task: freeze and implement the paper evaluation protocol (gw-flow-surrogate)

You are working in the `gw-flow-surrogate` repository (run everything from the repo root).
The model is a decoder-only autoregressive surrogate for NRHybSur3dq8 (2,2)-mode waveforms
with a flow-matching patch head, conditioned on θ = (q, χ1z, χ2z). Every number and figure in
the paper must come from ONE frozen, audited, reproducible evaluation protocol. Your job is to
audit the current evaluation code, fix it, build that protocol, verify it with tests, and run it.

Work in the phases below, in order. Do not skip the tests. Where this document says **STOP**,
write your report and wait for the human.

---

## Ground rules

1. **Never modify** training code paths' behaviour (`models/`, `datasets/`, `main.py`, `configs/`)
   or any checkpoint. Additive changes only (new scripts, new helper module, new CLI flags with
   backward-compatible defaults). If you think a training-side change is needed, report it.
2. **No silent fallbacks.** Any missing file, mismatched shape, or unexpected value must raise.
3. **No hard-coded result numbers** in any figure script. Figures read CSVs.
4. Every output CSV gets a sidecar `.json` with: checkpoint path + sha256, norm-stats path +
   sha256 + values, manifest path, split, row range, all protocol knobs, RNG seed, git commit
   (`git rev-parse HEAD` + dirty flag), torch version, GPU name, wall time.
5. Keep a running log in `results/protocol/WORKLOG.md`: what you ran, what you found, numbers.

---

## Known problems found in review (verify each one, cite file:line, fix as specified)

| # | Problem | Where | Required fix |
|---|---|---|---|
| P1 | **Norm stats are taken from the test manifest's folder**, not from the checkpoint's training data. A model evaluated on another dataset silently uses the wrong normalisation. | `eval_fm_unified.py` (`stats_path = Path(args.manifest).parent / "norm_stats.npz"`), `paper_figs/fig_calibration.py`, `paper_figs/fig_speed_frontier.py`, `scripts/measure_floor.py`, `scripts/run_prefreeze_pipeline.sh` (comment "norm_stats.npz must sit next to the manifest") | Norm stats are a property of the **checkpoint**. Resolve them from the run's `hparams.yaml` (`norm_stats_path`) or an explicit `--norm-stats`; raise if neither exists or the file is missing. |
| P2 | **Identity-stats fallback**: if the npz is missing, `WaveformPatchDataset._load_norm_stats` returns mean 0 / std 1 and evaluation runs on garbage without error. `evalute.py` guards against this; the other eval scripts do not. | `datasets/waveform_dataset.py::_load_norm_stats` | Do not change the dataset class. In every eval entry point, assert the stats file exists before constructing the dataset, and assert the loaded `ds.norm_stats` are not the identity. |
| P3 | **Ground truth is the token reconstruction, not the waveform.** `evaluate_batch` builds `h_true` from the true *tokens* (`_reconstruct_waveform(la_t, dp_t, res_t, …)`), so the representation floor (patch-constant amplitude) is excluded from every mismatch. | `evalute.py::evaluate_batch` | Headline mismatch = predicted vs **original** waveform `exp(log_a) · exp(-i(φ − φ[0]))` from the HDF5 tracks, in fp64, truncated to `n_patches·P` samples. Keep the token-space mismatch as a secondary column. Store the floor (true tokens vs original) per waveform. |
| P4 | `mismatch_strict_full` / `mismatch_phase_time_full` are **copies** of the `_future` columns, not full-waveform values. | `eval_fm_unified.py` (row dict) | Compute full-waveform mismatch for real: reconstruct `cat(true seed tokens, generated tokens)` as one sequence and compare with the original over all samples. |
| P5 | **No RNG seed** in any eval script; the model is stochastic, so reruns differ. | all eval scripts | `torch.manual_seed(seed + batch_index)` before each generation call; fixed batch size is part of the protocol. Same seed stream for every checkpoint (common random numbers). |
| P6 | `--n-steps` defaults to 20 in `eval_fm_unified.py`; README protocol says 40; training `n_sampling_steps` = 20. | `eval_fm_unified.py`, `README.md`, `results/prefreeze/PROTOCOL.txt` | `--n-steps` becomes a required argument, set from `PROTOCOL.yaml` (see Phase 2). |
| P7 | The checkpoint filename metric `val_mismatch` is **θ-only (context_patches: 1), phase-only, single sample (`rollout_mean_of=1`), mean over 4 waveforms**. It is a selection signal only. | `configs/scale_2p5m.yaml`, `models/lightning_module.py::_rollout_mismatch` | Never quote it. Checkpoint selection rule stays "lowest `val_mismatch` in filename" (document it); all reported numbers are re-evaluated. |
| P8 | `fig_q_anatomy.py` joins CSV rows to ground truth by **nearest q only** (`merge_asof` on q, tol 1e-4). Fragile. | `paper_figs/fig_q_anatomy.py::join` | Join on the test-row index (`idx`), which is identical across runs that use the same test rows. |
| P9 | `fig_scaling_final.py` hard-codes `CONVERGED_X/Y`. | `paper_figs/fig_scaling_final.py` | Read the per-run CSVs (Phase 4c) and compute everything. |
| P10 | `evaluate_batch` always runs a teacher-forced sampling pass (wasted compute at 5000 waveforms). | `evalute.py` | The new evaluator does not call `evaluate_batch`; it implements rollout + metrics directly (see Phase 2). |
| P11 | A newly generated test set must use **exactly** the training time grid (`t_min`, `t_max`, `dt`, peak alignment), otherwise patch positions and the absolute-time embedding no longer mean the same thing. | `data_generation/*` | Assert the new set's `t` array equals the training set's `t` array element-wise (`np.array_equal`). |

---

## Phase 0 — Audit (CPU only). Deliverable: `results/protocol/AUDIT.md`

1. **Norm-stats audit.** For every run under `scan_1M/n*/seed*/` (and any flagship run), read
   `hparams.yaml`: `norm_stats_path`, `manifest_path`, `residual_scale`, `patch_len`,
   `context_patches`, `n_sampling_steps`, `use_abs_time`. Load each npz, print its four arrays,
   and sha256 it. Write `results/protocol/norm_stats_audit.csv` (one row per run).
   - Expected: all scan runs share ONE frozen file (the scale-study design). If they do not
     (e.g. the 631k/1M runs used a different pool with different stats), list the groups. Each
     checkpoint must then be evaluated with **its own** stats; that is fine, as long as it is
     explicit.
   - If `hparams.yaml` does not contain `norm_stats_path`, find it from the training scripts
     (`scripts/scale_study/03_train_one.sh`, `PROTOCOL.env`) and record how you determined it.
2. **Checkpoint inventory.** For every run: list checkpoints, pick the one with the lowest
   `val_mismatch` in the filename (not `last.ckpt`), record it. Flag runs with no checkpoint
   (e.g. `n1000k/seed43`) and runs where the best epoch is suspiciously early (e.g. `n631k/seed44`,
   epoch 11/17); report but do not change the rule.
3. **Grid audit.** Record `n_time`, `dt`, `t[0]`, `t[-1]`, number of patches for the training
   pool(s) and every manifest you will evaluate on.
4. Confirm or refute each of P1–P11 with file:line references.

## Phase 1 — Test sets. Deliverable: `data/test5k/`

1. Generate a fresh evaluation set with **the same grid and parameter box as training** and a
   **seed never used for training** (check `PROTOCOL.env`, configs, generation logs):
   `python -m data_generation.cli generate --output-dir data/test5k --n-train 0 --n-test 5000
   --n-ood 1000 --seed <new> --dt <train dt> --t-min <train t_min> --t-max <train t_max> --jobs <k>`.
   If `--n-train 0` is not supported by `generate_dataset`, write a tiny wrapper that calls
   `uniform_parameters` / `extrapolation_parameters` and `_write_shard` directly. Do not edit
   the generator itself.
2. **Do not put a `norm_stats.npz` in `data/test5k/`.** Test data never defines normalisation.
3. Checks: `np.array_equal(t_test, t_train)`; all test θ inside [1,8]×[−0.8,0.8]²; minimum
   Euclidean distance (in normalised θ) from each test point to the training pool > 0 (report
   the distribution); no NaN/inf in tracks; amplitude peak at the same index for all rows.

## Phase 2 — Protocol file and one evaluator

### 2a. `results/protocol/PROTOCOL.yaml` (single source of truth)

```yaml
test_manifest: data/test5k/manifest.parquet
split: test
n_waveforms_flagship: 5000      # rows [0, 5000)
n_waveforms_scaling: 1000       # rows [0, 1000) -- same rows, a prefix
context_fraction: <TBD after Phase 4a>   # 0 = theta-only (1 patch)
n_steps: <TBD: knee of speed frontier>
mean_of: 4
temperature: 1.0
seed: 1234
batch_size: 64
headline_metric: mm_pt_gen       # median; see column spec
target_mismatch: [1.0e-3, 1.0e-4]
inference_dtype: fp32 network, fp64 reconstruction and mismatch
```

All eval scripts read this file (`--protocol results/protocol/PROTOCOL.yaml`); CLI flags may
override only for explicitly labelled ablations, and the override is recorded in the sidecar.

### 2b. Shared helpers: `paper_figs/protocol_utils.py`

Move into one module (and import it everywhere): `resolve_norm_stats(checkpoint, explicit)`,
`load_checkpoint(path)` (with the `use_abs_time=False` fallback), `reconstruct_fp64(...)`,
`mismatches(h1, h2) -> (strict, phase, phase_time, best_shift)`, `original_waveform(batch)`,
`write_sidecar(...)`, `load_protocol(path)`. `paper_figs/fig_context_fraction.py` already
contains reference implementations of the first four; refactor it to import them and prove
with a regression test that its outputs are unchanged (bitwise on CPU, ≤1e-12 relative on GPU).

### 2c. `eval_paper.py` (repo root): the only evaluator for paper numbers

For each test row it generates **once** with the protocol knobs and writes one CSV row with:

- identity: `idx, q, chi1z, chi2z, chi_eff, context_patches, t_seed_end_M`
- vs **original** waveform, generated segment: `mm_strict_gen, mm_phase_gen, mm_pt_gen, dt_shift_gen_M`
- vs original, full waveform (seed + generated): `mm_strict_full, mm_pt_full`
- vs **token reconstruction** (model error without floor): `mm_pt_gen_tokens`
- representation floor on the generated segment: `floor_pt_gen`
- dephasing: `dphi_peak_rad` (unwrapped φ_pred − φ_true at the amplitude peak, no alignment),
  `damp_peak_rel` (δA/A at the peak)

It also writes `<out>_tokens.npz` with the generated token features (float32, shape
`(n, n_gen, D)`) and the row indices, so Figure 2 / dephasing curves can be rebuilt later
without rerunning the model, plus `<out>_curves.npz` with Δφ(t) and δA/A(t) sampled at patch
boundaries for every row (for the dephasing figure).

Hard requirements: norm stats via `resolve_norm_stats` (P1/P2), original-waveform truth (P3),
real full-waveform columns (P4), seeding (P5), `n_steps` required (P6), no teacher-forced pass (P10).
Print a progress line with ETA. Refuse to run if the test manifest's `t` grid differs from the
training grid recorded in the audit (P11).

## Phase 3 — Correctness tests (`tests/test_eval_protocol.py`, all must pass)

Use synthetic data (see `scripts/make_smoke_cf.py`: chirp-like tracks, tiny untrained checkpoint
saved with `state_dict` + `hyper_parameters`) so the tests run on CPU in < 1 min.

1. **Round trip:** `reconstruct_fp64(true tokens)` vs original waveform has mismatch < 1e-3 on
   smooth synthetic chirps, and equals `GWFlowSurrogateLit._reconstruct_waveform` to 1e-6
   relative (fp32 vs fp64).
2. **Perfect prediction:** feeding the true tokens as "generated" gives `mm_*_tokens == 0`
   (to 1e-12) and `mm_*_gen == floor_*_gen`.
3. **Mismatch sanity:** identical waveforms → 0; a constant phase rotation → strict > 0, phase
   and phase_time = 0; a pure time shift by k samples → phase_time ≈ 0 and `best_shift == k`.
4. **Norm-stats guard:** missing file raises; identity stats raise; `hparams.yaml` resolution works.
5. **Determinism:** two runs with the same seed and batch size give identical CSVs.
6. **Grid guard:** a manifest with a different `dt` raises.
7. **Real-data regression (GPU, run once, not in CI):** before any change, run the *old*
   `eval_fm_unified.py` on the 1M seed-42 checkpoint with the settings that produced the current
   scaling figure (ctx 0.5, 500 rows of the original test manifest, same `n_steps`). Record the
   median. Then run `eval_paper.py` with the same knobs; `mm_pt_gen_tokens` median must agree
   within statistical noise (±5%), because that column is what the old script computed. If it
   does not agree, **STOP** and report.

## Phase 4 — Runs

**4a. Context fraction (decides the protocol).** Flagship = `scan_1M/n1000k/seed42` best checkpoint.
```
python paper_figs/fig_context_fraction.py eval --checkpoint <flagship> \
  --manifest data/test5k/manifest.parquet --split test --n-waveforms 1000 \
  --fractions 0 0.25 0.5 0.75 --n-steps <current best guess, 40> --mean-of 4 --seed 1234 \
  --out results/context_fraction.csv
python paper_figs/fig_context_fraction.py plot results/context_fraction.csv
```
Also run it for one small-N checkpoint (e.g. n63k seed42) to see whether the θ-only gap closes with data.
**STOP**: report the summary table (median / p90 / p99 / % below 1e-3 on the common window, per
fraction, plus floor) and the figure. The human sets `context_fraction` in `PROTOCOL.yaml`.

**4b. Speed frontier → `n_steps`.** Rerun `fig_speed_frontier.py` with the chosen context,
`mean_of=4`, seeding and correct norm stats (port it to `protocol_utils`). Timing: CUDA-synchronised,
2 warm-up batches, batch 1 **and** batch 64, report ms/waveform and waveforms/s; time NRHybSur3dq8
on the same machine at the same `dt` and grid. **STOP**: the human picks the knee `n_steps`.

**4c. Flagship evaluation (5000 waveforms):** `eval_paper.py --protocol ... --checkpoint <flagship>
--n-waveforms 5000 --out results/paper/flagship_1M_s42.csv`. Also the `ood` split (1000 rows).
Before launching, time 64 waveforms and report the projected wall time.

**4d. Scaling re-evaluation:** every best checkpoint in `scan_1M` (≈30 runs) on rows [0, 1000) with
the identical protocol → `results/paper/scaling/n{N}_s{seed}.csv`. Then `results/paper/scaling_summary.csv`:
per run the median of `mm_pt_gen` (and `mm_pt_gen_tokens`, `floor_pt_gen`); per N the geometric
mean over seeds, min, max, n_seeds. Rewrite `fig_scaling_final.py` to read this file; single-seed N
gets an open marker. Fit the power law (a) on all N and (b) on N ≥ 60k only; report both slopes with
uncertainties and N* for 1e-4 and 5e-5. Draw the median representation floor as a horizontal line.

**4e. UQ:** S = 8 independent single-sample rollouts (`mean_of=1`) on rows [0, 1000) of the
flagship; spread = mean pairwise phase+time mismatch between samples; **true error = `mm_pt_gen`
of the same row from the 4c mean-of-4 run** (join on `idx`). Same seeds rule.

**4f. Floor:** `floor_pt_gen` already comes from 4c; additionally report the floor vs `patch_len`
∈ {8, 16, 32, 64} (data only, no model) in `results/paper/floor_vs_patchlen.csv`.

## Phase 5 — Figures

Every figure script reads only the CSVs/npz from Phase 4 and `PROTOCOL.yaml`, writes PDF + PNG via
`paper_figs/style.py`, and prints its caption facts (n, protocol knobs, checkpoint). Port
`fig_calibration.py`, `fig_speed_frontier.py`, `fig_q_anatomy.py` (join on `idx`, P8),
`fig_scaling_by_q.py`, `fig_scaling_final.py` (P9) accordingly. New: a binned (q, χ_eff) heatmap
(median per cell, cells with < 20 waveforms masked, counts shown) and the mismatch CDF, both from
the flagship CSV.

## Final report: `results/protocol/REPORT.md`

1. Audit findings (P1–P11: confirmed/refuted, file:line, fix commit).
2. Test results (all 7).
3. Regression check (old vs new number).
4. Headline table for the flagship: median, p90, p99, max, % < 1e-3, % < 1e-4 for `mm_pt_gen`,
   `mm_strict_gen`, `mm_pt_full`, `mm_pt_gen_tokens`, `floor_pt_gen`, median |Δφ_peak|, median |Δt|.
   Each with a bootstrap 68% CI of the median.
5. Scaling table and both fits.
6. Every command run, with wall time and GPU.
7. Anything surprising, and anything you could not verify.
