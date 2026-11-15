#!/usr/bin/env bash
# Step 1 — generate the 300k waveform pool once (~10 CPU-hours).
#
# Usage (from repo root):
#   bash scripts/scale_study/01_generate.sh
#
# Or on SLURM:
#   bash scripts/scale_study/slurm/submit_generate.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "Generating ${N_TRAIN_POOL} train waveforms → ${DATA_DIR}"
log "Python: ${PYTHON}"

if [[ -f "${DATA_DIR}/manifest.parquet" ]]; then
  warn "Parent manifest already exists: ${DATA_DIR}/manifest.parquet"
  warn "Delete it (and HDF5 shards) if you really want to regenerate."
  exit 0
fi

py -m data_generation.cli generate \
  --output-dir "${DATA_DIR}" \
  --n-files 1 \
  --n-train "${N_TRAIN_POOL}" \
  --n-test "${N_TEST}" \
  --n-ood "${N_OOD}" \
  --val-fraction "${VAL_FRACTION}" \
  --dt "${DT}" \
  --seed "${SEED}" \
  --q-min "${Q_MIN}" \
  --q-max "${Q_MAX}" \
  --chi-min "${CHI_MIN}" \
  --chi-max "${CHI_MAX}"

need_file "${DATA_DIR}/manifest.parquet"
log "Done. Next: bash scripts/scale_study/02_freeze_and_slice.sh"
