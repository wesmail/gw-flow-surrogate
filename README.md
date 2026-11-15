# gw-flow-surrogate

PyTorch Lightning **decoder-only autoregressive surrogate** for quasi-circular, spin-aligned BBH gravitational waveforms. The model predicts amplitude–phase patch tokens with a **flow matching head**, conditioned on $\theta = (q, \chi_{1z}, \chi_{2z})$.

Ground truth comes from **NRHybSur3dq8** via `gwsurrogate` (geometric units, $G=c=1$).

---

## Quick map

| Step | What |
|------|------|
| **0** | Install packages (below) |
| **A–E** | **Scaling-law figure** — [Exact steps after install](#exact-steps-after-install-scaling-law-figure) |
| **F** | Optional: flagship 1M train without nested subsets |

Paper numbers use the frozen protocol in that section (θ-only, pchip, `n_steps=40`, `mean_of=4`). Do not use ad-hoc `evalute.py` defaults.

Older README: [`archive/README_old_scaling.md`](archive/README_old_scaling.md).

---

## 0. Required packages and install

| Package | Role |
|---------|------|
| `torch`, `lightning`, `jsonargparse[signatures]` | Training / CLI |
| `tensorboard` / optional `wandb` | Logging |
| `numpy`, `scipy`, `h5py`, `pyarrow`, `pandas`, `matplotlib`, `tqdm` | Data + eval |
| `gwsurrogate` | **Data generation only** (NRHybSur3dq8) |
| `pytest` | Tests (optional) |

```bash
cd gw-flow-surrogate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# gwsurrogate often needs GSL — e.g. conda install -c conda-forge gsl
# or an IGWN env (igwn-py311) for generation
```

You can train/eval on an already generated pool without `gwsurrogate`.

Optional: `python smoke_test.py` and `pip install pytest && pytest tests/ -v`.

---

## Exact steps after install (scaling-law figure)

Do these in order. Goal: six nested sizes × seeds → CSVs → one figure.

**What you are reproducing.** One 1M training pool (Sobol, seed 42). Nested $N \in \{16\mathrm{k},\,40\mathrm{k},\,100\mathrm{k},\,251\mathrm{k},\,631\mathrm{k},\,1\mathrm{M}\}$ = first $N$ train points. Same val/test for every $N$. Same flagship YAML, fixed **320 333** steps. Headline: median **`mm_pt_full`**.

### A. One-time machine setup

```bash
# 1. Edit cluster/env.sh: uncomment module load + activate your venv/conda
#    with torch (+ gwsurrogate if you will generate). Set POOL if not data/n1000k.

# 2. In every cluster/*.sbatch: uncomment #SBATCH --partition / --account

# 3. Put the two small frozen reference files in place (shipped separately or
#    copied from an existing pool). They must match the paper pool.
mkdir -p data/ref
cp /path/to/n1000k_manifest.parquet data/ref/n1000k_manifest.parquet
cp /path/to/norm_stats.npz          data/ref/norm_stats.npz
# If this machine already has the full pool:
#   cp data/n1000k/manifest.parquet data/ref/n1000k_manifest.parquet
#   cp data/n1000k/norm_stats.npz   data/ref/norm_stats.npz
```

### B. Download NRHybSur3dq8 and check the plan

```bash
python scripts/gen_shard.py download
python scripts/gen_shard.py plan \
  --infer-from data/ref/n1000k_manifest.parquet \
  --reference  data/ref/n1000k_manifest.parquet
# must print: PLAN MATCHES THE REFERENCE
```

### C. Generate the pool (~180 GB, 64 shards)

**SLURM:**

```bash
mkdir -p logs/slurm
sbatch --array=0-63 cluster/gen_pool.sbatch
# wait until all 64 finish; re-submit is safe (done shards are skipped)
bash cluster/finalize_pool.sh
# must print: POOL MATCHES THE REFERENCE
```

**No SLURM** (same scripts):

```bash
mkdir -p logs/slurm
for i in $(seq 0 63); do SLURM_ARRAY_TASK_ID=$i bash cluster/gen_pool.sbatch; done
bash cluster/finalize_pool.sh
```

`finalize_pool.sh` merges manifests, copies **frozen** `norm_stats.npz` from `data/ref/`, and writes `manifest_train_<N>.parquet` for each size.

**Skip C** if `data/n1000k/` already has `manifest.parquet`, `norm_stats.npz`, shards, and the nested `manifest_train_*.parquet` files. Then only ensure `data/ref/` matches (step A.3) and jump to D.

### D. Train + eval (writes one CSV per run)

Each job trains to `step=320333.ckpt`, then evals 1000 test rows → `results/scaling/N<N>_seed<S>.csv`.

**SLURM — phase 1 (6 sizes, seed 0):**

```bash
bash cluster/submit_scaling.sh
```

**SLURM — phase 2 (more seeds):**

```bash
SEEDS="1 2" bash cluster/submit_scaling.sh
```

**No SLURM** (example: one size):

```bash
N=16000 SEED=0 bash cluster/train_scaling.sbatch
# repeat for every N and seed you need
```

Cost: ~**15 GPU-h** per run on an RTX 4090. Phase 1 = 6 runs; three seeds × six $N$ = 18 runs.

### E. Aggregate + plot

```bash
python scripts/make_scaling_summary.py \
  --glob 'results/scaling/N*_seed*.csv' \
  --out results/paper/scaling_summary.csv

python paper_figs/fig_scaling_final.py \
  --summary results/paper/scaling_summary.csv
# writes paper_figs/out/fm_scaling_final.{pdf,png} (via style.save)
```

### What success looks like

| Path | Meaning |
|------|---------|
| `logs/scaling/N<N>/seed<S>/checkpoints/step=320333.ckpt` | Final model (always this step; not best-val) |
| `results/scaling/N<N>_seed<S>.csv` | Per-waveform mismatches; use column **`mm_pt_full`** |
| `results/paper/scaling_summary.csv` | Per-seed + per-$N$ medians for the figure |
| `paper_figs/out/fm_scaling_final.*` | Scaling-law figure |

### Frozen choices (do not change)

| Choice | Value |
|--------|--------|
| Norms | `data/ref/norm_stats.npz` → copied into the pool; **never** recomputed |
| Val / ckpt interval | every **7813** steps (`configs/scaling_overrides.yaml`) |
| Eval | θ-only (`--fractions 0`), `pchip`, `--n-steps 40`, `--mean-of 4`, `--seed 1234` |
| Test rows | first **1000** of pool `test` |
| Headline | median **`mm_pt_full`**; also % $<10^{-3}$ and same for $\chi_\mathrm{eff}>0.4$ |
| Model YAML | `configs/n1000k_2p5m_final.yaml` |

Pool layout: `--n-train 1000000 --val-fraction 0.1 --n-test 5000 --n-ood 500 --n-files 64 --seed 42` → **1 105 500** waveforms.

### Troubleshooting

| Symptom | What to do |
|---------|------------|
| Shard missing | Resubmit array; finished shards are skipped |
| Job killed | Resubmit / chain; resumes from `last.ckpt` |
| `finalize` won’t recompute norms | Intended — norms are frozen from `REF_NORM` |
| Plan / compare fails | `GEN_ARGS` or seed changed mid-generation |
| No CSVs for the plot | Wait until `train_scaling` prints `done: ... -> results/scaling/...csv` |

Helpers: [`cluster/`](cluster/) (that folder’s README only points here).

---

## Flagship training (full 1M pool)

Same recipe as the scaling runs, without nested subsets:

```bash
python main.py fit --config configs/n1000k_2p5m_final.yaml \
  --data.manifest_path ./data/n1000k/manifest.parquet \
  --data.norm_stats_path ./data/n1000k/norm_stats.npz
```

(`configs/scale_2p5m.yaml` is the same recipe under the older filename.)

Monitor with TensorBoard (`logs/`) or override to W&B via LightningCLI if desired.

---

## Evaluate one checkpoint (manual)

Frozen protocol (matches the scaling jobs):

```bash
CKPT=logs/scaling/N1000000/seed0/checkpoints/step=320333.ckpt
POOL=data/n1000k

python paper_figs/fig_context_fraction.py eval \
  --checkpoint "$CKPT" \
  --manifest "$POOL/manifest.parquet" --norm-stats "$POOL/norm_stats.npz" \
  --split test --n-waveforms 1000 --fractions 0 \
  --n-steps 40 --mean-of 4 --seed 1234 --amp-mode pchip \
  --out results/scaling/manual_eval.csv
```

Headline: median of `mm_pt_full`. Also report fraction $<10^{-3}$ and the same for $\chi_\mathrm{eff}>0.4$.

---

## Repository layout

```
gw-flow-surrogate/
├── main.py
├── smoke_test.py
├── eval_fm_unified.py / evalute.py
├── configs/
│   ├── n1000k_2p5m_final.yaml   # flagship recipe
│   └── scaling_overrides.yaml   # fixed-step val/ckpt for scaling
├── cluster/                     # SLURM + env for the scaling study
├── scripts/gen_shard.py
├── scripts/make_subset_manifests.py
├── scripts/make_scaling_summary.py
├── data_generation/  datasets/  models/  paper_figs/  tests/
├── archive/                     # old docs / scale_study / prefreeze helpers
├── data/                        # gitignored — pool + shards
└── logs/                        # gitignored — training
```

---

## Design notes

- Absolute phase is gauge; the model predicts increments + de-ramped residuals.
- `log_a` / `Δφ`: train-split patch norm (`norm_stats.npz`); residuals ÷ `residual_scale`.
- Causal AR: position $p$ attends to $\le p$; flow targets future horizons.

---

## Citation

Neural surrogate of **NRHybSur3dq8** (Varma et al.). Ground-truth waveforms are surrogate/NR replicas within that model’s domain, not independent Einstein-equation solutions.
