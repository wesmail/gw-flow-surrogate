#!/usr/bin/env bash
#SBATCH --job-name=gw_scale_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Internal SLURM body — prefer submit_generate.sh which sets resources from PROTOCOL.env.
set -euo pipefail

# When submitted via sbatch --export, ROOT / SCALE_PROTOCOL are available.
# Fallback: discover from this script location.
if [[ -z "${ROOT:-}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
export SCALE_PROTOCOL="${SCALE_PROTOCOL:-${ROOT}/scripts/scale_study/PROTOCOL.env}"

echo "Host=$(hostname)  ROOT=${ROOT}  DATE=$(date -Is)"
cd "${ROOT}"
bash "${ROOT}/scripts/scale_study/01_generate.sh"
echo "Generate finished at $(date -Is)"
