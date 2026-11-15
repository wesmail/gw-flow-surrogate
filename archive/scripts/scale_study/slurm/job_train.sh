#!/usr/bin/env bash
#SBATCH --job-name=gw_scale_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Body for one training run. RUN_ID is passed via --export or as $1.
# Prefer submit_train_array.sh.
set -euo pipefail

if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
export SCALE_PROTOCOL="${SCALE_PROTOCOL:-${ROOT}/scripts/scale_study/PROTOCOL.env}"

# Array job: map SLURM_ARRAY_TASK_ID → run_id via a phase-filtered list.
RUN_ID="${RUN_ID:-${1:-}}"
if [[ -z "${RUN_ID}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  LIST_FILE="${RUN_LIST_FILE:-${ROOT}/results/scale_study/slurm_run_list.txt}"
  RUN_ID="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${LIST_FILE}")"
fi
[[ -n "${RUN_ID}" ]] || { echo "RUN_ID not set"; exit 1; }

echo "Host=$(hostname)  RUN_ID=${RUN_ID}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}  DATE=$(date -Is)"
cd "${ROOT}"
bash "${ROOT}/scripts/scale_study/03_train_one.sh" "${RUN_ID}"
echo "Train ${RUN_ID} finished at $(date -Is)"
