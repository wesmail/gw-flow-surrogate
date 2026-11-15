#!/bin/bash
# Submit the scaling grid: every N x every SEED, each as a chain of CHAIN jobs
# (job k+1 starts after job k ends and resumes from last.ckpt if needed).
#
#   bash cluster/submit_scaling.sh                       # phase 1: 6 sizes, seed 0
#   SEEDS="1 2" bash cluster/submit_scaling.sh           # phase 2: two more seeds
#   SIZES="100000 1000000" SEEDS="1 2" bash cluster/submit_scaling.sh   # only some N
#   DRY=1 bash cluster/submit_scaling.sh                 # print, submit nothing
set -euo pipefail
cd "$(dirname "$0")/.."

SIZES=(${SIZES:-16000 40000 100000 251000 631000 1000000})
SEEDS=(${SEEDS:-0})
CHAIN="${CHAIN:-2}"          # 24 h jobs per run; 2 covers a slow GPU or queue kills

source cluster/env.sh
for N in "${SIZES[@]}"; do
  [[ -f "$POOL/manifest_train_${N}.parquet" ]] || { echo "missing $POOL/manifest_train_${N}.parquet"; exit 1; }
done
mkdir -p logs/slurm

for N in "${SIZES[@]}"; do
  for S in "${SEEDS[@]}"; do
    name="gw_N${N}_s${S}"
    cmd=(sbatch --parsable --job-name "$name" --export "ALL,N=$N,SEED=$S" cluster/train_scaling.sbatch)
    if [[ -n "${DRY:-}" ]]; then echo "${cmd[*]}  (x$CHAIN, chained)"; continue; fi
    jid=$("${cmd[@]}")
    ids="$jid"
    for ((k = 2; k <= CHAIN; k++)); do
      jid=$(sbatch --parsable --dependency="afterany:$jid" --job-name "$name" \
                   --export "ALL,N=$N,SEED=$S" cluster/train_scaling.sbatch)
      ids="$ids -> $jid"
    done
    echo "N=$N seed=$S : $ids"
  done
done
