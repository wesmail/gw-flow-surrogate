#!/bin/bash
# One array task = one shard. 64 shards -> 64 independent CPU jobs.
#   sbatch --array=0-63 cluster/gen_pool.sbatch
# Re-submitting is safe: finished shards are skipped.
#SBATCH --job-name=gw_gen
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster/env.sh

python scripts/gen_shard.py shard --infer-from "$REF_MANIFEST" --output-dir "$POOL" \
    --index "${SLURM_ARRAY_TASK_ID:?run as an array job}"
