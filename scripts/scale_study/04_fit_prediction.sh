#!/usr/bin/env bash
# Fit power law on scaling points and WRITE prediction.json (gate before 300k).
#
# Usage (after eval CSVs exist for 6k…120k @ 2.5M):
#   bash scripts/scale_study/04_fit_prediction.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

log "Fitting + writing prediction.json gate"
py scripts/scale_study/04_eval_and_fit.py \
  --results-dir "${RESULTS_DIR}" \
  --runs-tsv "${RUNS_TSV}" \
  --mismatch-col "${MISMATCH_COL:-mismatch_phase_time_future}" \
  --target-mismatch "${TARGET_MISMATCH:-1e-4}" \
  --predict-at-n 300000 \
  --fit-phases scaling \
  --write-prediction

need_file "${RESULTS_DIR}/prediction.json"
log "Headline numbers:"
py -c "import json; h=json.load(open('${RESULTS_DIR}/prediction.json'))['headline'];
print('  predicted mismatch @ 300k:', h['predicted_mismatch_at_300k']);
print('  predicted N for 1e-4:    ', h['predicted_n_for_1e-4']);
print('  (with floor) @ 300k:     ', h['predicted_mismatch_at_300k_with_floor']);
print('  (with floor) N for 1e-4: ', h['predicted_n_for_1e-4_with_floor'])"

warn "Write these two numbers down. THEN train/eval 300k_2p5m."
warn "  bash scripts/scale_study/03_train_one.sh 300k_2p5m"
