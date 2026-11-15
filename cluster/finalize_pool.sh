#!/bin/bash
# After all generation shards finished: merge the 64 shard manifests, check the
# pool equals the laptop pool, install the FROZEN norm stats, write the nested
# training subsets. A few minutes on CPU; run on the login node or via sbatch.
#   bash cluster/finalize_pool.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source cluster/env.sh
SIZES=(${SIZES:-16000 40000 100000 251000 631000 1000000})

n_h5=$(ls "$POOL"/waveforms_*.manifest.parquet 2>/dev/null | wc -l)
echo "shard manifests found: $n_h5"

python -m data_generation.cli combine --output-dir "$POOL"          # -> $POOL/manifest.parquet
python scripts/gen_shard.py compare --manifest "$POOL/manifest.parquet" --reference "$REF_MANIFEST"

# frozen normalisation: the laptop file, never recomputed
cp -v "$REF_NORM" "$POOL/norm_stats.npz"
python - "$POOL/norm_stats.npz" <<'EOF'
import sys, numpy as np
s = np.load(sys.argv[1])
print({k: np.round(s[k], 6).tolist() for k in s.files})
assert not (np.allclose(s["log_a_mean"], 0) and np.allclose(s["log_a_std"], 1)), "identity stats!"
EOF

python scripts/make_subset_manifests.py --manifest "$POOL/manifest.parquet" --sizes "${SIZES[@]}"
echo "pool ready: $POOL"
