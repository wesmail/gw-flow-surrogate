#!/usr/bin/env bash
# Submit the one-shot 300k data generation job (CPU).
#
# Usage:
#   bash scripts/scale_study/slurm/submit_generate.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

ACCOUNT_ARGS=()
[[ -n "${SLURM_ACCOUNT:-}" ]] && ACCOUNT_ARGS+=(--account="${SLURM_ACCOUNT}")
MAIL_ARGS=()
[[ -n "${SLURM_MAIL_USER:-}" ]] && MAIL_ARGS+=(--mail-user="${SLURM_MAIL_USER}" --mail-type=END,FAIL)

mkdir -p "${RESULTS_DIR}/slurm_logs"

sbatch \
  "${ACCOUNT_ARGS[@]}" \
  "${MAIL_ARGS[@]}" \
  --job-name=gw_scale_gen \
  --partition="${SLURM_PARTITION_CPU}" \
  --time="${SLURM_TIME_GENERATE}" \
  --cpus-per-task="${SLURM_CPUS_GENERATE}" \
  --mem="${SLURM_MEM_GENERATE}" \
  --output="${RESULTS_DIR}/slurm_logs/generate-%j.out" \
  --error="${RESULTS_DIR}/slurm_logs/generate-%j.err" \
  --export=ALL,ROOT="${ROOT}",SCALE_PROTOCOL="${ROOT}/scripts/scale_study/PROTOCOL.env" \
  "${ROOT}/scripts/scale_study/slurm/job_generate.sh"

log "Submitted generate job. After it finishes:"
log "  bash scripts/scale_study/02_freeze_and_slice.sh"
