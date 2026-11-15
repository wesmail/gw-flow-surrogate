#!/usr/bin/env bash
# Step 2 — freeze norm stats + residual_scale; carve nested train manifests.
#
# Usage:
#   bash scripts/scale_study/02_freeze_and_slice.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

need_file "${DATA_DIR}/manifest.parquet"

# shellcheck disable=SC2206
SIZES=( ${TRAIN_SIZES:-6000 12000 25000 50000 120000 300000} )

log "Freezing norms + slicing manifests: ${SIZES[*]}"
py scripts/scale_study/02_freeze_and_slice.py \
  --data-dir "${DATA_DIR}" \
  --sizes "${SIZES[@]}" \
  --results-dir "${RESULTS_DIR}" \
  --force

need_file "${FROZEN_JSON}"
need_file "${DATA_DIR}/norm_stats.npz"
log "Frozen protocol → ${FROZEN_JSON}"
log "residual_scale = $(residual_scale_from_frozen)"
log "Next: train scaling runs, e.g."
log "  bash scripts/scale_study/03_train_one.sh 6k_2p5m"
log "  bash scripts/scale_study/03_train_grid.sh scaling"
log "Or SLURM: bash scripts/scale_study/slurm/submit_train_array.sh scaling"
