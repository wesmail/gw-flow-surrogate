#!/usr/bin/env bash
# Submit a SLURM array over scale-study training runs.
#
# Usage:
#   bash scripts/scale_study/slurm/submit_train_array.sh              # scaling+capacity
#   bash scripts/scale_study/slurm/submit_train_array.sh scaling
#   bash scripts/scale_study/slurm/submit_train_array.sh capacity
#   bash scripts/scale_study/slurm/submit_train_array.sh heldout       # needs prediction.json
#   bash scripts/scale_study/slurm/submit_train_array.sh all
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

FILTER="${1:-default}"
need_file "${FROZEN_JSON}"
mkdir -p "${RESULTS_DIR}/slurm_logs"

LIST_FILE="${RESULTS_DIR}/slurm_run_list.txt"
: > "${LIST_FILE}"

while IFS=$'\t' read -r run_id n_train config model_tag phase notes; do
  [[ "${run_id}" == "run_id" ]] && continue
  [[ -z "${run_id}" || "${run_id}" =~ ^# ]] && continue
  case "${FILTER}" in
    default) [[ "${phase}" == "heldout" ]] && continue ;;
    scaling|capacity|heldout) [[ "${phase}" == "${FILTER}" ]] || continue ;;
    all) ;;
    *) die "Unknown filter: ${FILTER}" ;;
  esac
  if [[ "${phase}" == "heldout" ]]; then
    need_file "${RESULTS_DIR}/prediction.json"
  fi
  echo "${run_id}" >> "${LIST_FILE}"
done < "${RUNS_TSV}"

N="$(wc -l < "${LIST_FILE}" | tr -d ' ')"
[[ "${N}" -ge 1 ]] || die "No runs matched filter=${FILTER}"
log "Submitting ${N} train jobs (filter=${FILTER}):"
cat "${LIST_FILE}" | sed 's/^/  /'

ACCOUNT_ARGS=()
[[ -n "${SLURM_ACCOUNT:-}" ]] && ACCOUNT_ARGS+=(--account="${SLURM_ACCOUNT}")
MAIL_ARGS=()
[[ -n "${SLURM_MAIL_USER:-}" ]] && MAIL_ARGS+=(--mail-user="${SLURM_MAIL_USER}" --mail-type=END,FAIL)

# SLURM arrays are 0-indexed or 1-indexed depending on site; we use 1..N.
sbatch \
  "${ACCOUNT_ARGS[@]}" \
  "${MAIL_ARGS[@]}" \
  --job-name=gw_scale_train \
  --partition="${SLURM_PARTITION_GPU}" \
  --gres="${SLURM_GPU_GRES}" \
  --time="${SLURM_TIME_TRAIN}" \
  --cpus-per-task="${SLURM_CPUS_TRAIN}" \
  --mem="${SLURM_MEM_TRAIN}" \
  --array=1-"${N}" \
  --output="${RESULTS_DIR}/slurm_logs/train-%A_%a.out" \
  --error="${RESULTS_DIR}/slurm_logs/train-%A_%a.err" \
  --export=ALL,ROOT="${ROOT}",SCALE_PROTOCOL="${ROOT}/scripts/scale_study/PROTOCOL.env",RUN_LIST_FILE="${LIST_FILE}" \
  "${ROOT}/scripts/scale_study/slurm/job_train.sh"

log "After all train jobs finish, eval with:"
log "  bash scripts/scale_study/slurm/submit_eval_array.sh ${FILTER}"
