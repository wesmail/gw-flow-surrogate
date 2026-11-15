#!/usr/bin/env bash
# Train a subset of runs sequentially (local / single GPU).
#
# Usage:
#   bash scripts/scale_study/03_train_grid.sh              # all except heldout
#   bash scripts/scale_study/03_train_grid.sh scaling      # 6k…120k @ 2.5M
#   bash scripts/scale_study/03_train_grid.sh capacity     # 0.6M + 10M @ 120k
#   bash scripts/scale_study/03_train_grid.sh heldout      # 300k (needs prediction.json)
#   bash scripts/scale_study/03_train_grid.sh all          # everything (heldout gated)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FILTER="${1:-default}"
SCALE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log "Train grid filter=${FILTER}"

while IFS=$'\t' read -r run_id n_train config model_tag phase notes; do
  [[ "${run_id}" == "run_id" ]] && continue
  [[ -z "${run_id}" || "${run_id}" =~ ^# ]] && continue

  case "${FILTER}" in
    default)
      # Everything except heldout — write prediction before 300k.
      [[ "${phase}" == "heldout" ]] && continue
      ;;
    scaling|capacity|heldout)
      [[ "${phase}" == "${FILTER}" ]] || continue
      ;;
    all) ;;
    *)
      die "Unknown filter: ${FILTER} (use scaling|capacity|heldout|all|default)"
      ;;
  esac

  log "=== ${run_id} (phase=${phase}) ==="
  bash "${SCALE_DIR}/03_train_one.sh" "${run_id}"
done < "${RUNS_TSV}"

log "Grid done (filter=${FILTER})."
