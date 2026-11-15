#!/usr/bin/env bash
# =============================================================================
# run_prefreeze_pipeline.sh — gated paper/talk evaluation pipeline
# =============================================================================
#
# WHAT THIS IS
# ------------
# A *phased* runner for the post-training path that produces paper/talk numbers.
# It is intentionally NOT one blind end-to-end command: Phases A and D have
# human gates (regression check; capacity vs corner-coverage decision). Running
# past a failed gate can lock wrong n_steps / wrong CSVs into every quoted
# number before the Aug 14 freeze.
#
# WHERE TO RUN
# ------------
# Always from the repository root:
#   cd /path/to/GWSurrogate
#   bash scripts/run_prefreeze_pipeline.sh help
#
# DEPENDENCY ORDER (do not reorder)
# ---------------------------------
#   A  regression check          → prove code still matches prior numbers
#   B  freeze eval protocol      → speed frontier picks --n-steps; write PROTOCOL
#   C  new measurements          → floor / e2e / corrected TF ratio
#   D  scaling-by-q decision     → GATE: capacity run or not
#   E  remaining figures         → calibration, q-anatomy, overlays, plots
#   F  text checklist            → PAPER_NOTES paste (no heavy compute)
#
# QUICK START
# -----------
#   1. Edit the CONFIG block below (CKPT, MANIFEST, PYTHON, CSV paths).
#   2. bash scripts/run_prefreeze_pipeline.sh phase_a
#   3. If A looks good:  bash scripts/run_prefreeze_pipeline.sh phase_b
#   4. Set N_STEPS to the knee, then:  bash scripts/run_prefreeze_pipeline.sh freeze_protocol
#   5. Continue phase_c → phase_d → (gate) → phase_e → phase_f
#
# Optional: regenerate scaling CSVs under the frozen protocol if n_steps changed:
#   bash scripts/run_prefreeze_pipeline.sh regen_scaling_csvs
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# CONFIG — edit these once; every phase reuses them
# -----------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Interpreter (pytorch env for eval; igwn-py311 only needed for data gen / GSL)
PYTHON="${PYTHON:-$HOME/micromamba/envs/pytorch/bin/python}"

# Paper checkpoint + data (norm_stats.npz must sit next to the manifest)
CKPT="${CKPT:-scaling/logs/flow_25k_small/version_1/checkpoints/epoch=163-val_mismatch=0.0060.ckpt}"
MANIFEST="${MANIFEST:-data/25k/manifest.parquet}"
SPLIT="${SPLIT:-test}"

# Frozen protocol fields (Phase B fills N_STEPS; do not change the rest for
# any number you intend to quote in the paper/talk)
CONTEXT_FRACTION="${CONTEXT_FRACTION:-0.5}"
N_WAVEFORMS="${N_WAVEFORMS:-500}"
N_SAMPLES="${N_SAMPLES:-1}"
MEAN_OF="${MEAN_OF:-4}"
# [GATE] Set after phase_b by inspecting results/speed_frontier.csv (knee).
# Default 20 matches many existing scaling CSVs; change only after Phase B.
N_STEPS="${N_STEPS:-20}"

# Scaling-study per-waveform CSVs (best of each pair; 12k excluded — not converged)
CSV_6K="${CSV_6K:-results/scale_version_0_epoch=449.csv}"
CSV_25K="${CSV_25K:-results/scale_version_1_epoch=163.csv}"
CSV_120K="${CSV_120K:-results/fm_120k_e092.csv}"

# Optional: paths to a *previous* metrics.json / CSV for Phase A diffs
# Leave empty to only print new metrics (you compare by eye / git).
BASELINE_METRICS="${BASELINE_METRICS:-}"
BASELINE_UNIFIED_CSV="${BASELINE_UNIFIED_CSV:-}"

# Output dirs
OUT_A="${OUT_A:-results/prefreeze/phase_a}"
OUT_PROTO="${OUT_PROTO:-results/prefreeze/PROTOCOL.txt}"
DEVICE="${DEVICE:-}"   # empty → let each script pick cuda if available

mkdir -p results/prefreeze paper_figs/out eval_test "$OUT_A"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gate]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

need_file() {
  [[ -f "$1" ]] || die "Missing file: $1
Set the path in the CONFIG block at the top of this script."
}

py() {
  # shellcheck disable=SC2086
  "$PYTHON" "$@"
}

device_flag() {
  if [[ -n "$DEVICE" ]]; then
    echo "--device $DEVICE"
  fi
}

print_protocol() {
  cat <<EOF
FROZEN EVAL PROTOCOL (quote every paper/talk number under this, or regenerate)
  split=$SPLIT
  context-fraction=$CONTEXT_FRACTION
  n-waveforms=$N_WAVEFORMS
  n-samples=$N_SAMPLES
  mean-of=$MEAN_OF
  n-steps=$N_STEPS
  checkpoint=$CKPT
  manifest=$MANIFEST
EOF
}

# =============================================================================
# PHASE A — regression check
# =============================================================================
# Exact purpose:
#   Prove that the prefreeze code changes did not silently move old numbers.
#   - smoke_test.py: unit-style correctness (KV cache, loss, reconstruct).
#   - evalute.py: diagnostics protocol (default context_patches=1). New keys
#     context_patches / context_fraction / n_generated_patches must appear;
#     other metrics should match an old metrics.json up to sampling noise.
#   - eval_fm_unified.py: paper CSV path. ITEM1 TF-parity does NOT touch this
#     script — CSV should match a prior run (identical protocol ⇒ identical
#     columns within noise).
#
# How to run:
#   bash scripts/run_prefreeze_pipeline.sh phase_a
#
# What you must do by hand after:
#   Diff metrics.json vs BASELINE_METRICS (if set) / an old eval_test run.
#   Diff the unified CSV vs BASELINE_UNIFIED_CSV.
#   If anything shifted beyond noise → STOP. Do not enter Phase B.
# =============================================================================
phase_a() {
  need_file "$CKPT"
  need_file "$MANIFEST"
  need_file "$PYTHON"

  log "Phase A.1 — smoke_test.py (CPU, ~1 min)"
  py smoke_test.py

  log "Phase A.2 — evalute.py regression (default context=1; short for speed)"
  # Uses a reduced n-waveforms for a quick check; for a strict bit-check against
  # a full prior run, bump --n-waveforms to match that run.
  # shellcheck disable=SC2046
  py evalute.py \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --n-waveforms 32 \
    --n-samples 1 \
    --n-steps "$N_STEPS" \
    --mean-of 1 \
    --n-plot 0 \
    --output-dir "$OUT_A/evalute_default" \
    $(device_flag)

  log "Phase A.2 keys (expect context_patches=1, context_fraction=null, n_generated=319)"
  METRICS_A="$OUT_A/evalute_default/metrics.json" py - <<'PY'
import json, os
from pathlib import Path
m = json.loads(Path(os.environ["METRICS_A"]).read_text())
for k in ("context_patches", "context_fraction", "n_generated_patches",
          "tf_to_rollout_dphi_ratio", "mismatch_mean"):
    print(f"  {k}: {m.get(k)}")
PY

  if [[ -n "$BASELINE_METRICS" && -f "$BASELINE_METRICS" ]]; then
    log "Diff vs BASELINE_METRICS=$BASELINE_METRICS (ignore new context_* keys)"
    py - "$BASELINE_METRICS" "$OUT_A/evalute_default/metrics.json" <<'PY'
import json, sys
old, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
skip = {"context_patches", "context_fraction", "n_generated_patches"}
for k in sorted(set(old) | set(new)):
    if k in skip: continue
    if old.get(k) != new.get(k):
        print(f"  CHANGED {k}:\n    old={old.get(k)}\n    new={new.get(k)}")
PY
  else
    warn "BASELINE_METRICS unset — compare $OUT_A/evalute_default/metrics.json to an older run by hand."
  fi

  log "Phase A.3 — eval_fm_unified.py regression (ctx 0.5; short for speed)"
  py eval_fm_unified.py \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --context-fraction "$CONTEXT_FRACTION" \
    --n-waveforms 32 \
    --n-samples "$N_SAMPLES" \
    --mean-of "$MEAN_OF" \
    --n-steps "$N_STEPS" \
    --out "$OUT_A/unified_smoke.csv" \
    --device "${DEVICE:-cpu}"

  if [[ -n "$BASELINE_UNIFIED_CSV" && -f "$BASELINE_UNIFIED_CSV" ]]; then
    log "Median mismatch vs BASELINE_UNIFIED_CSV"
    py - "$BASELINE_UNIFIED_CSV" "$OUT_A/unified_smoke.csv" <<'PY'
import sys, pandas as pd, numpy as np
col = "mismatch_phase_time_future"
a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
# Smoke run may be shorter — compare overlapping prefix
n = min(len(a), len(b))
print(f"  baseline p50={np.median(a[col].iloc[:n]):.6e}  new p50={np.median(b[col].iloc[:n]):.6e}  (n={n})")
PY
  else
    warn "BASELINE_UNIFIED_CSV unset — compare $OUT_A/unified_smoke.csv to an older CSV by hand."
  fi

  warn "GATE A: If metrics/CSV shifted beyond sampling noise, STOP and investigate."
  warn "If OK → proceed to:  bash scripts/run_prefreeze_pipeline.sh phase_b"
}

# =============================================================================
# PHASE B — freeze evaluation protocol (speed frontier first)
# =============================================================================
# Exact purpose:
#   The flow head costs n_sampling_steps Euler evaluations per patch. Too few
#   steps → worse mismatch; too many → wasted wall-clock. Phase B sweeps
#   n_steps ∈ {5,10,20,40,100}, you pick the *knee*, then freeze the full
#   protocol line used for every quoted number.
#
# How to run:
#   bash scripts/run_prefreeze_pipeline.sh phase_b
#   # inspect results/speed_frontier.csv — pick knee (e.g. 40 if >> better than 20)
#   N_STEPS=40 bash scripts/run_prefreeze_pipeline.sh freeze_protocol
#
# If the chosen N_STEPS differs from what existing scaling CSVs used, regenerate:
#   N_STEPS=40 bash scripts/run_prefreeze_pipeline.sh regen_scaling_csvs
# =============================================================================
phase_b() {
  need_file "$CKPT"
  need_file "$MANIFEST"

  log "Phase B — speed frontier EVAL (§6.1): steps 5 10 20 40 100"
  warn "This is GPU-heavy (or slow on CPU). Decision rule: pick the knee;"
  warn "if 40 is meaningfully better than 20, use 40 for all final numbers."

  py paper_figs/fig_speed_frontier.py eval \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --context-fraction "$CONTEXT_FRACTION" \
    --steps 5 10 20 40 100 \
    --n-waveforms 200 \
    --out results/speed_frontier.csv

  log "Wrote results/speed_frontier.csv — inspect mismatch vs n_steps:"
  py - <<'PY'
import pandas as pd
d = pd.read_csv("results/speed_frontier.csv")
print(d.to_string(index=False))
PY

  warn "GATE B: Choose N_STEPS, then run:"
  warn "  N_STEPS=<knee> bash scripts/run_prefreeze_pipeline.sh freeze_protocol"
  warn "Optional plot (after choosing):"
  warn "  bash scripts/run_prefreeze_pipeline.sh phase_e_speed_plot"
}

freeze_protocol() {
  print_protocol | tee "$OUT_PROTO"
  log "Wrote $OUT_PROTO — keep this next to every paper table."
  warn "If N_STEPS differs from old scaling CSVs, run: regen_scaling_csvs"
}

# Regenerate the three converged scaling CSVs under the frozen protocol.
# Exact purpose: consistency across the study beats any single-point improvement.
regen_scaling_csvs() {
  need_file "$CKPT"
  need_file "$MANIFEST"
  print_protocol

  # Map: label -> checkpoint (edit if your "best of pair" differs)
  declare -A CKPTS=(
    [6k]="scaling/logs/flow_6k_small/version_0/checkpoints/epoch=449-val_mismatch=0.0041.ckpt"
    [25k]="scaling/logs/flow_25k_small/version_1/checkpoints/epoch=163-val_mismatch=0.0060.ckpt"
  )
  # 120k: set CKPT_120K if you have it locally; otherwise skip and keep CSV_120K
  CKPT_120K="${CKPT_120K:-}"

  for label in 6k 25k; do
    c="${CKPTS[$label]}"
    need_file "$c"
    out="results/scale_${label}_nsteps${N_STEPS}_ctx${CONTEXT_FRACTION}.csv"
    log "Regenerating $label → $out"
    py eval_fm_unified.py \
      --checkpoint "$c" \
      --manifest "$MANIFEST" \
      --split "$SPLIT" \
      --context-fraction "$CONTEXT_FRACTION" \
      --n-waveforms "$N_WAVEFORMS" \
      --n-samples "$N_SAMPLES" \
      --mean-of "$MEAN_OF" \
      --n-steps "$N_STEPS" \
      --out "$out" \
      --device "${DEVICE:-cpu}"
  done

  if [[ -n "$CKPT_120K" && -f "$CKPT_120K" ]]; then
    out="results/scale_120k_nsteps${N_STEPS}_ctx${CONTEXT_FRACTION}.csv"
    log "Regenerating 120k → $out"
    py eval_fm_unified.py \
      --checkpoint "$CKPT_120K" \
      --manifest "$MANIFEST" \
      --split "$SPLIT" \
      --context-fraction "$CONTEXT_FRACTION" \
      --n-waveforms "$N_WAVEFORMS" \
      --n-samples "$N_SAMPLES" \
      --mean-of "$MEAN_OF" \
      --n-steps "$N_STEPS" \
      --out "$out" \
      --device "${DEVICE:-cpu}"
    warn "Update CSV_120K=$out in this script's CONFIG for Phase D/E."
  else
    warn "CKPT_120K unset — keeping existing $CSV_120K (ensure same n_steps!)."
  fi
}

# =============================================================================
# PHASE C — three new measurements
# =============================================================================
# Exact purpose:
#   (1) Representation floor on real data (no model) — O(1e-4), roughly q-flat.
#   (2) Three-way table with checkpoint: floor / token-space / end-to-end.
#       End-to-end is what a referee reads as the headline "vs NRHybSur" number.
#   (3) Corrected tf_to_rollout_dphi_ratio with --mean-of 4 (ITEM1 parity).
#
# How to run:
#   N_STEPS=<frozen> bash scripts/run_prefreeze_pipeline.sh phase_c
# =============================================================================
phase_c() {
  need_file "$CKPT"
  need_file "$MANIFEST"
  print_protocol

  log "Phase C.1 — measure_floor.py floor-only (n=$N_WAVEFORMS)"
  py scripts/measure_floor.py \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --n-waveforms "$N_WAVEFORMS" \
    --out results/prefreeze/floor_only.json

  log "Phase C.2 — floor + end-to-end + token-space (same n; GPU recommended)"
  py scripts/measure_floor.py \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --n-waveforms "$N_WAVEFORMS" \
    --checkpoint "$CKPT" \
    --n-steps "$N_STEPS" \
    --device "${DEVICE:-cpu}" \
    --out results/prefreeze/floor_e2e.json

  log "Phase C.3 — evalute.py with --mean-of 4 (corrected TF/rollout ratio)"
  # shellcheck disable=SC2046
  py evalute.py \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --n-waveforms 128 \
    --n-samples 1 \
    --n-steps "$N_STEPS" \
    --mean-of 4 \
    --temperature 1 \
    --n-plot 0 \
    --output-dir results/prefreeze/evalute_meanof4 \
    $(device_flag)

  py - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("results/prefreeze/evalute_meanof4/metrics.json").read_text())
print("tf_to_rollout_dphi_ratio =", m["tf_to_rollout_dphi_ratio"])
print("Update the draft wherever the OLD (pre-ITEM1) ratio appears.")
f = json.loads(Path("results/prefreeze/floor_e2e.json").read_text())
print("Three-way medians:")
for k in ("floor", "token_space", "end_to_end"):
    v = f.get(k)
    if isinstance(v, dict):
        print(f"  {k}: median={v.get('median')}  p90={v.get('p90')}")
    else:
        print(f"  {k}: {v}")
PY
}

# =============================================================================
# PHASE D — scaling-by-q decision (GATE)
# =============================================================================
# Exact purpose:
#   The global scaling slope bends. Per-q-bin slopes distinguish:
#     corner coverage → edge bins flatten, interior keeps ~-1.1 → NO capacity run
#     capacity limit  → slopes shallow uniformly → launch 2×-wide 120k by ~Aug 3–5
#   Decide the day you see the plot — not later.
#
# How to run:
#   bash scripts/run_prefreeze_pipeline.sh phase_d
# =============================================================================
phase_d() {
  need_file "$CSV_6K"
  need_file "$CSV_25K"
  need_file "$CSV_120K"

  log "Phase D — fig_scaling_by_q.py (12k deliberately excluded)"
  py paper_figs/fig_scaling_by_q.py \
    --points "6000:$CSV_6K" "25000:$CSV_25K" "120000:$CSV_120K" \
    --out-csv results/scaling_by_q.csv

  log "Per-bin slopes are printed above and in results/scaling_by_q.csv"
  warn "GATE D:"
  warn "  Edge flattens, interior ~-1.1 → corner coverage; figure is the result; skip capacity run."
  warn "  Slopes shallow everywhere → capacity live; calendar: 2x-wide 120k by ~Aug 3–5."
  warn "If OK to proceed to figures:  bash scripts/run_prefreeze_pipeline.sh phase_e"
}

# =============================================================================
# PHASE E — remaining figures (eval-only)
# =============================================================================
# Exact purpose:
#   Produce talk/paper figures under the frozen protocol:
#   calibration (S=8), q-anatomy, scaling_final cross-check, speed plot,
#   waveform overlays with drift panel.
#
# How to run:
#   bash scripts/run_prefreeze_pipeline.sh phase_e          # all of E
#   bash scripts/run_prefreeze_pipeline.sh phase_e_calib    # only calibration eval+plot
#   bash scripts/run_prefreeze_pipeline.sh phase_e_speed_plot
# =============================================================================
phase_e_calib() {
  need_file "$CKPT"
  need_file "$MANIFEST"
  print_protocol

  log "Phase E — calibration EVAL (500 wf × S=8; long)"
  py paper_figs/fig_calibration.py eval \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --context-fraction "$CONTEXT_FRACTION" \
    --n-waveforms "$N_WAVEFORMS" \
    --S 8 \
    --n-steps "$N_STEPS" \
    --out results/calibration.csv

  log "Phase E — calibration PLOT"
  py paper_figs/fig_calibration.py plot results/calibration.csv
  warn "Check Spearman (spread vs error). If spread predicts error → talk slide."
}

phase_e_speed_plot() {
  need_file results/speed_frontier.csv
  log "Phase E — speed frontier PLOT (from Phase B CSV)"
  py paper_figs/fig_speed_frontier.py plot results/speed_frontier.csv
}

phase_e() {
  phase_e_calib

  log "Phase E — q-anatomy (25k vs 120k CSVs)"
  need_file "$CSV_25K"
  need_file "$CSV_120K"
  py paper_figs/fig_q_anatomy.py \
    --csv-25k "$CSV_25K" \
    --csv-120k "$CSV_120K" \
    --manifest "$MANIFEST" \
    --context-fraction "$CONTEXT_FRACTION"

  log "Phase E — fig_scaling_final.py (cross-check embedded numbers by hand!)"
  py paper_figs/fig_scaling_final.py
  warn "Compare embedded CONVERGED_Y in fig_scaling_final.py to medians of CSV_6K/25K/120K."

  phase_e_speed_plot

  log "Phase E — evalute.py overlays at frozen protocol (--n-plot 4)"
  # Prefer ctx-fraction matching paper numbers for overlays used next to tables.
  # shellcheck disable=SC2046
  py evalute.py \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --context-fraction "$CONTEXT_FRACTION" \
    --n-waveforms 32 \
    --n-samples 1 \
    --n-steps "$N_STEPS" \
    --mean-of "$MEAN_OF" \
    --n-plot 4 \
    --output-dir results/prefreeze/overlays \
    $(device_flag)

  log "Overlays → results/prefreeze/overlays/waveform_*.png"
}

# =============================================================================
# PHASE F — text checklist (no heavy compute)
# =============================================================================
# Exact purpose:
#   Paste PAPER_NOTES_prefreeze.md into methods by ~Aug 11. No GPU.
#
# How to run:
#   bash scripts/run_prefreeze_pipeline.sh phase_f
# =============================================================================
phase_f() {
  need_file PAPER_NOTES_prefreeze.md
  need_file "$OUT_PROTO"
  log "Phase F — text checklist (open these files and paste into the draft)"
  echo "  1. PAPER_NOTES_prefreeze.md  → metric name, (2,2) degeneracy, param counts"
  echo "  2. $OUT_PROTO                → state context protocol next to every table"
  echo "  3. results/prefreeze/floor_e2e.json → floor / token-space / end-to-end (one sentence each)"
  echo "  4. Headline tables: ONE mismatch column (phase+time-optimised only)"
  echo "  5. Talk slides: scaling(+per-bin), q-anatomy, overlay+drift, calibration, speed"
  echo
  cat "$OUT_PROTO"
  echo
  warn "Do NOT implement anything on the DO-NOT list in PAPER_NOTES_prefreeze.md before freeze."
}

# =============================================================================
# help / dispatch
# =============================================================================
usage() {
  cat <<EOF
Usage: bash scripts/run_prefreeze_pipeline.sh <command>

Commands (run in order; stop at gates):
  phase_a              Regression: smoke_test + evalute + eval_fm_unified
  phase_b              Speed-frontier sweep → choose N_STEPS
  freeze_protocol      Write results/prefreeze/PROTOCOL.txt (set N_STEPS=...)
  regen_scaling_csvs   Re-eval 6k/25k/(120k) under frozen protocol
  phase_c              Floor / e2e / corrected tf_to_rollout ratio
  phase_d              Per-q scaling plot + capacity GATE
  phase_e              All remaining figures + overlays
  phase_e_calib        Calibration eval+plot only
  phase_e_speed_plot   Speed frontier plot only
  phase_f              Text paste checklist
  show_protocol        Print current CONFIG protocol
  help                 This message

Environment overrides (examples):
  CKPT=path/to.ckpt N_STEPS=40 DEVICE=cuda bash scripts/run_prefreeze_pipeline.sh phase_c
  PYTHON=\$HOME/micromamba/envs/pytorch/bin/python bash scripts/run_prefreeze_pipeline.sh phase_a

See README.md § "Paper / talk evaluation pipeline (pre-freeze)".
EOF
}

cmd="${1:-help}"
case "$cmd" in
  phase_a)             phase_a ;;
  phase_b)             phase_b ;;
  freeze_protocol)     freeze_protocol ;;
  regen_scaling_csvs)  regen_scaling_csvs ;;
  phase_c)             phase_c ;;
  phase_d)             phase_d ;;
  phase_e)             phase_e ;;
  phase_e_calib)       phase_e_calib ;;
  phase_e_speed_plot)  phase_e_speed_plot ;;
  phase_f)             phase_f ;;
  show_protocol)       print_protocol ;;
  help|-h|--help)      usage ;;
  *) die "Unknown command: $cmd (try: help)" ;;
esac
