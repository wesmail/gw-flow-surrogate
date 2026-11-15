#!/usr/bin/env bash
# Evaluate one run with the frozen paper protocol → results/scale_study/eval_<id>.csv
#
# Usage:
#   bash scripts/scale_study/04_eval_one.sh 6k_2p5m
#   CKPT=/path/to.ckpt bash scripts/scale_study/04_eval_one.sh 6k_2p5m
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_ID="${1:-}"
[[ -n "${RUN_ID}" ]] || die "Usage: $0 <run_id>"

ROW="$(lookup_run "${RUN_ID}")" || die "Unknown run_id: ${RUN_ID}"
N_TRAIN="$(echo "${ROW}" | cut -f2)"
PHASE="$(echo "${ROW}" | cut -f5)"

require_prediction_before_heldout "${PHASE}"

MANIFEST="$(manifest_for_n "${N_TRAIN}")"
need_file "${MANIFEST}"
# eval_fm_unified resolves norm_stats next to the manifest — ensure frozen file is there.
need_file "${DATA_DIR}/norm_stats.npz"

CKPT="${CKPT:-$(best_checkpoint "${RUN_ID}")}"
need_file "${CKPT}"

OUT="${RESULTS_DIR}/eval_${RUN_ID}.csv"
log "Eval ${RUN_ID}"
log "  ckpt=${CKPT}"
log "  manifest=${MANIFEST}"
log "  n_steps=${N_STEPS}  mean_of=${MEAN_OF}  n_waveforms=${N_WAVEFORMS}"

py eval_fm_unified.py \
  --checkpoint "${CKPT}" \
  --manifest "${MANIFEST}" \
  --split "${EVAL_SPLIT:-test}" \
  --context-fraction "${CONTEXT_FRACTION:-0.5}" \
  --n-waveforms "${N_WAVEFORMS:-500}" \
  --n-samples "${N_SAMPLES:-1}" \
  --mean-of "${MEAN_OF:-4}" \
  --n-steps "${N_STEPS:-20}" \
  --out "${OUT}" \
  --device "${DEVICE:-cuda}"

log "Wrote ${OUT}"
