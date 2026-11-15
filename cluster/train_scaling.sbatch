#!/bin/bash
# Train ONE scaling run (N, SEED) for the fixed 320,333-step schedule, then
# evaluate its final checkpoint with the frozen protocol.
#   sbatch --export=ALL,N=100000,SEED=0 cluster/train_scaling.sbatch
# Safe to submit again (or chain with --dependency=afterany):
#   * final checkpoint + CSV exist  -> exits immediately
#   * last.ckpt exists              -> resumes from it (loses < 7813 steps)
#SBATCH --job-name=gw_scale
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cluster/env.sh

: "${N:?set N, e.g. --export=ALL,N=100000,SEED=0}"
SEED="${SEED:-0}"
BASE_CONFIG="${BASE_CONFIG:-configs/n1000k_2p5m_final.yaml}"
OVERRIDES="${OVERRIDES:-configs/scaling_overrides.yaml}"
FINAL_STEP="${FINAL_STEP:-320333}"               # must equal trainer.max_steps
EVAL_ROWS="${EVAL_ROWS:-1000}"

RUN_DIR="logs/scaling/N${N}/seed${SEED}"
CKPT_DIR="$RUN_DIR/checkpoints"
FINAL="$CKPT_DIR/$(printf 'step=%06d.ckpt' "$FINAL_STEP")"
CSV="results/scaling/N${N}_seed${SEED}.csv"
mkdir -p logs/slurm results/scaling

echo "run N=$N seed=$SEED  host=$(hostname)  $(date)"

# ---- 1. train (or resume) ------------------------------------------------
if [[ ! -f "$FINAL" ]]; then
  RESUME=()
  if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
    RESUME=(--ckpt_path "$CKPT_DIR/last.ckpt")
    echo "resuming from $CKPT_DIR/last.ckpt"
  fi
  python main.py fit --config "$BASE_CONFIG" --config "$OVERRIDES" \
      --seed_everything "$SEED" \
      --data.manifest_path "$POOL/manifest_train_${N}.parquet" \
      --data.norm_stats_path "$POOL/norm_stats.npz" \
      --trainer.logger.init_args.name "N${N}" \
      --trainer.logger.init_args.version "seed${SEED}" \
      ${RESUME[@]+"${RESUME[@]}"}
fi
[[ -f "$FINAL" ]] || { echo "final checkpoint $FINAL missing after fit"; exit 1; }

# ---- 2. evaluate with the frozen protocol (theta-only, pchip) -------------
if [[ ! -f "$CSV" ]]; then
  python paper_figs/fig_context_fraction.py eval --checkpoint "$FINAL" \
      --manifest "$POOL/manifest.parquet" --norm-stats "$POOL/norm_stats.npz" \
      --split test --n-waveforms "$EVAL_ROWS" --fractions 0 \
      --n-steps 40 --mean-of 4 --seed 1234 --amp-mode pchip \
      --out "$CSV"
fi
echo "done: $FINAL  ->  $CSV   $(date)"
