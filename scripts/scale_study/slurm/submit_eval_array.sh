#!/usr/bin/env bash
# Submit a SLURM array to evaluate trained runs.
#
# Usage:
#   bash scripts/scale_study/slurm/submit_eval_array.sh scaling
#   bash scripts/scale_study/slurm/submit_eval_array.sh capacity
#   bash scripts/scale_study/slurm/submit_eval_array.sh heldout
#   bash scripts/scale_study/slurm/submit_eval_array.sh all
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

FILTER="${1:-scaling}"
mkdir -p "${RESULTS_DIR}/slurm_logs"

LIST_FILE="${RESULTS_DIR}/slurm_eval_list.txt"
: > "${LIST_FILE}"

while IFS=$'\t' read -r run_id n_train config model_tag phase notes; do
  [[ "${run_id}" == "run_id" ]] && continue
  [[ -z "${run_id}" || "${run_id}" =~ ^# ]] && continue
  case "${FILTER}" in
    scaling|capacity|heldout) [[ "${phase}" == "${FILTER}" ]] || continue ;;
    all|default) ;;
    *) die "Unknown filter: ${FILTER}" ;;
  esac
  echo "${run_id}" >> "${LIST_FILE}"
done < "${RUNS_TSV}"

N="$(wc -l < "${LIST_FILE}" | tr -d ' ')"
[[ "${N}" -ge 1 ]] || die "No runs matched filter=${FILTER}"
log "Submitting ${N} eval jobs (filter=${FILTER}):"
cat "${LIST_FILE}" | sed 's/^/  /'

ACCOUNT_ARGS=()
[[ -n "${SLURM_ACCOUNT:-}" ]] && ACCOUNT_ARGS+=(--account="${SLURM_ACCOUNT}")
MAIL_ARGS=()
[[ -n "${SLURM_MAIL_USER:-}" ]] && MAIL_ARGS+=(--mail-user="${SLURM_MAIL_USER}" --mail-type=END,FAIL)

sbatch \
  "${ACCOUNT_ARGS[@]}" \
  "${MAIL_ARGS[@]}" \
  --job-name=gw_scale_eval \
  --partition="${SLURM_PARTITION_GPU}" \
  --gres="${SLURM_GPU_GRES}" \
  --time="${SLURM_TIME_EVAL}" \
  --cpus-per-task=4 \
  --mem=32G \
  --array=1-"${N}" \
  --output="${RESULTS_DIR}/slurm_logs/eval-%A_%a.out" \
  --error="${RESULTS_DIR}/slurm_logs/eval-%A_%a.err" \
  --export=ALL,ROOT="${ROOT}",SCALE_PROTOCOL="${ROOT}/scripts/scale_study/PROTOCOL.env",RUN_LIST_FILE="${LIST_FILE}" \
  "${ROOT}/scripts/scale_study/slurm/job_eval.sh"

log "When evals finish:"
log "  bash scripts/scale_study/04_fit_prediction.sh     # after scaling evals"
log "  python scripts/scale_study/05_plot_scaling.py"
