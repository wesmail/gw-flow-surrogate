# Sourced by every job script. EDIT ONCE for your cluster.
# Loads Python + CUDA and activates the environment that has torch/lightning
# (and gwsurrogate for data generation).

# module load cuda/12.1            # example: whatever your cluster uses
# source ~/micromamba/etc/profile.d/micromamba.sh && micromamba activate pytorch
# source ~/venvs/gw/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Where the pool lives (shards + manifests + norm_stats.npz). Large: ~180 GB
# for 1.1M waveforms. Point it at the cluster's work/scratch filesystem.
export POOL="${POOL:-data/n1000k}"
# Copies of the two laptop files (small): reference manifest + frozen norm stats
export REF_MANIFEST="${REF_MANIFEST:-data/ref/n1000k_manifest.parquet}"
export REF_NORM="${REF_NORM:-data/ref/norm_stats.npz}"
