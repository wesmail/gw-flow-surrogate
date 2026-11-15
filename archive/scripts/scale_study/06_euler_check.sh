#!/usr/bin/env bash
# Euler-step sensitivity check (plan step 6 / 7): 20, 50, 100 steps.
#
# Usage:
#   bash scripts/scale_study/06_euler_check.sh 120k_2p5m
#   bash scripts/scale_study/06_euler_check.sh 300k_2p5m
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_ID="${1:-}"
[[ -n "${RUN_ID}" ]] || die "Usage: $0 <run_id>"

ROW="$(lookup_run "${RUN_ID}")" || die "Unknown run_id: ${RUN_ID}"
N_TRAIN="$(echo "${ROW}" | cut -f2)"
MANIFEST="$(manifest_for_n "${N_TRAIN}")"
CKPT="${CKPT:-$(best_checkpoint "${RUN_ID}")}"
need_file "${CKPT}"
need_file "${MANIFEST}"

OUT="${RESULTS_DIR}/speed_frontier_${RUN_ID}.csv"
log "Euler-step sweep for ${RUN_ID} → ${OUT}"

py paper_figs/fig_speed_frontier.py eval \
  --checkpoint "${CKPT}" \
  --manifest "${MANIFEST}" \
  --split "${EVAL_SPLIT:-test}" \
  --context-fraction "${CONTEXT_FRACTION:-0.5}" \
  --steps 20 50 100 \
  --n-waveforms "${N_WAVEFORMS_EULER:-200}" \
  --out "${OUT}"

log "Inspect ${OUT}. If 50/100 clearly beat 20, re-eval the study with the knee n_steps."
