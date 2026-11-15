#!/usr/bin/env bash
# Train a single scale-study run by run_id (from runs.tsv).
#
# Usage:
#   bash scripts/scale_study/03_train_one.sh 6k_2p5m
#   bash scripts/scale_study/03_train_one.sh 120k_10m
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_ID="${1:-}"
[[ -n "${RUN_ID}" ]] || die "Usage: $0 <run_id>"

ROW="$(lookup_run "${RUN_ID}")" || die "Unknown run_id: ${RUN_ID} (see runs.tsv)"
# run_id n_train config model_tag phase notes
N_TRAIN="$(echo "${ROW}" | cut -f2)"
CONFIG="$(echo "${ROW}" | cut -f3)"
MODEL_TAG="$(echo "${ROW}" | cut -f4)"
PHASE="$(echo "${ROW}" | cut -f5)"

require_prediction_before_heldout "${PHASE}"

need_file "${FROZEN_JSON}"
need_file "${CONFIG}"
MANIFEST="$(manifest_for_n "${N_TRAIN}")"
need_file "${MANIFEST}"
need_file "${DATA_DIR}/norm_stats.npz"

RESIDUAL_SCALE="$(residual_scale_from_frozen)"
NORM_STATS="${DATA_DIR}/norm_stats.npz"

log "Train ${RUN_ID}"
log "  config=${CONFIG}  N=${N_TRAIN}  model=${MODEL_TAG}  phase=${PHASE}"
log "  manifest=${MANIFEST}"
log "  residual_scale=${RESIDUAL_SCALE}"
log "  norm_stats=${NORM_STATS}"

py main.py fit --config "${CONFIG}" \
  --seed_everything "${SEED:-42}" \
  --trainer.max_epochs "${MAX_EPOCHS:-500}" \
  --trainer.logger.init_args.save_dir "${LOG_DIR}" \
  --trainer.logger.init_args.name "${RUN_ID}" \
  --data.manifest_path "${MANIFEST}" \
  --data.norm_stats_path "${NORM_STATS}" \
  --data.compute_stats_if_missing false \
  --model.residual_scale "${RESIDUAL_SCALE}"

log "Done ${RUN_ID}. Checkpoint under ${LOG_DIR}/${RUN_ID}/"
log "Eval: bash scripts/scale_study/04_eval_one.sh ${RUN_ID}"
