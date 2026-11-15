#!/usr/bin/env bash
# Evaluate a subset of runs, then optionally fit the power law.
#
# Usage:
#   bash scripts/scale_study/04_eval_grid.sh scaling --fit
#   bash scripts/scale_study/04_eval_grid.sh capacity
#   bash scripts/scale_study/04_eval_grid.sh all
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FILTER="${1:-scaling}"
DO_FIT=0
shift || true
for arg in "$@"; do
  case "${arg}" in
    --fit) DO_FIT=1 ;;
    --write-prediction) DO_FIT=1; WRITE_PRED=1 ;;
  esac
done
WRITE_PRED="${WRITE_PRED:-0}"
SCALE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while IFS=$'\t' read -r run_id n_train config model_tag phase notes; do
  [[ "${run_id}" == "run_id" ]] && continue
  [[ -z "${run_id}" || "${run_id}" =~ ^# ]] && continue
  case "${FILTER}" in
    scaling|capacity|heldout) [[ "${phase}" == "${FILTER}" ]] || continue ;;
    all|default) ;;
    *) die "Unknown filter: ${FILTER}" ;;
  esac
  bash "${SCALE_DIR}/04_eval_one.sh" "${run_id}"
done < "${RUNS_TSV}"

if [[ "${DO_FIT}" -eq 1 ]]; then
  log "Fitting power law (phases=scaling)"
  EXTRA=()
  if [[ "${WRITE_PRED}" -eq 1 ]]; then
    EXTRA+=(--write-prediction)
  fi
  py scripts/scale_study/04_eval_and_fit.py \
    --results-dir "${RESULTS_DIR}" \
    --runs-tsv "${RUNS_TSV}" \
    --mismatch-col "${MISMATCH_COL:-mismatch_phase_time_future}" \
    --target-mismatch "${TARGET_MISMATCH:-1e-4}" \
    --fit-phases scaling \
    "${EXTRA[@]}"
fi
