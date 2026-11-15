#!/usr/bin/env bash
#SBATCH --job-name=gw_scale_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
export SCALE_PROTOCOL="${SCALE_PROTOCOL:-${ROOT}/scripts/scale_study/PROTOCOL.env}"

RUN_ID="${RUN_ID:-${1:-}}"
if [[ -z "${RUN_ID}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  LIST_FILE="${RUN_LIST_FILE:-${ROOT}/results/scale_study/slurm_run_list.txt}"
  RUN_ID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${LIST_FILE}")"
fi
[[ -n "${RUN_ID}" ]] || { echo "RUN_ID not set"; exit 1; }

echo "Host=$(hostname)  RUN_ID=${RUN_ID}  DATE=$(date -Is)"
cd "${ROOT}"
bash "${ROOT}/scripts/scale_study/04_eval_one.sh" "${RUN_ID}"
echo "Eval ${RUN_ID} finished at $(date -Is)"
