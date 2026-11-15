# Data-scaling study

**Goal.** Find how many training waveforms are needed for a median mismatch of $10^{-4}$, and whether the ~2.5M-parameter model is large enough to get there.

This folder turns that plan into a **phased, tick-off pipeline**: generate once → freeze norms → train eight runs → fit a power law → decide → (optional) target run.

You can run everything **locally** (bash) or on a **SLURM cluster** (array jobs). Same scripts underneath.

---

## Quick map

| Step | What | Local command | SLURM |
|------|------|---------------|-------|
| 0 | Copy protocol + edit paths | `cp PROTOCOL.env.example PROTOCOL.env` | same |
| 1 | Generate 300k pool (~10 CPU-h) | `bash 01_generate.sh` | `bash slurm/submit_generate.sh` |
| 2 | Freeze norms + nested manifests | `bash 02_freeze_and_slice.sh` | same (short CPU) |
| 3a | Train scaling grid (6k…120k @ 2.5M) | `bash 03_train_grid.sh scaling` | `bash slurm/submit_train_array.sh scaling` |
| 3b | Train capacity (0.6M + 10M @ 120k) | `bash 03_train_grid.sh capacity` | `bash slurm/submit_train_array.sh capacity` |
| 4a | Eval scaling runs | `bash 04_eval_grid.sh scaling` | `bash slurm/submit_eval_array.sh scaling` |
| 4b | **Write prediction** (gate) | `bash 04_fit_prediction.sh` | same |
| 5 | Train + eval 300k held-out | `03_train_one.sh` / `04_eval_one.sh 300k_2p5m` | `submit_*_array.sh heldout` |
| 6 | Decide | fill `DECISION.md` | — |
| 7 | Plot | `python 05_plot_scaling.py` | — |
| 8 | Euler check (if floor suspect) | `bash 06_euler_check.sh <run_id>` | — |

Eight trainings before the target run: five data sizes × 2.5M, plus 0.6M and 10M at 120k, plus 300k @ 2.5M as the held-out check.

---

## Design rules (do not break these)

1. **One waveform pool.** Generate 300k train waveforms once (Sobol, $q\in[1,8]$, $|\chi|\le 0.8$). Smaller sets are the **first $N$ train rows** of that pool — nested by construction.
2. **Same val / test for every run.** Subset manifests keep the full val, test, and ood rows from the parent.
3. **Norm stats frozen once.** `norm_stats.npz` is computed on the **full** 300k train split and reused everywhere (`compute_stats_if_missing=false`). This rules out the “120k norm-stats bug” by design.
4. **Same `residual_scale`** for every run (written to `results/scale_study/frozen.json`).
5. **Same hyperparameters + EarlyStopping** (`patience=25` on `val_mismatch`, `seed=42`).
6. **Same eval** for every quoted number: test split, `context-fraction=0.5`, `n-waveforms=500`, `n-samples=1`, `mean-of=4`, `n-steps=20` (override in `PROTOCOL.env` only after an Euler check).
7. **Prediction before 300k.** `04_fit_prediction.sh` writes `prediction.json`. Training `300k_2p5m` refuses to start until that file exists.

---

## Setup (once)

```bash
cd /path/to/gw-flow-surrogate

# 1. Protocol
cp scripts/scale_study/PROTOCOL.env.example scripts/scale_study/PROTOCOL.env
# Edit PROTOCOL.env:
#   PYTHON=...          # e.g. $HOME/micromamba/envs/pytorch/bin/python
#   DATA_DIR=data/scale300k
#   SLURM_*             # if using a cluster

# 2. Envs
# Data generation needs gwsurrogate (+ GSL). Training/eval need pytorch + lightning.
# On many clusters: igwn-py311 for generate, pytorch env for train/eval.
```

Configs used by this study:

| File | Approx params | Role |
|------|---------------|------|
| `configs/scale_2p5m.yaml` | ~3.1M (paper: “2.5M”) | Data-scaling grid + 300k check |
| `configs/scale_0p6m.yaml` | ~0.65M | Capacity probe at 120k |
| `configs/scale_10m.yaml` | ~9.6M | Capacity probe at 120k |

Run catalog: `scripts/scale_study/runs.tsv`.

---

## Step-by-step

### 1. Generate the pool

```bash
bash scripts/scale_study/01_generate.sh
# SLURM:
bash scripts/scale_study/slurm/submit_generate.sh
```

Produces `data/scale300k/manifest.parquet` + `waveforms_*.h5`.

**Note.** Generation currently samples **val** from the same scrambled Sobol stream as train (disjoint continuation), and **test** uniformly. The plan prefers a uniform val set; if you need that, say so and we can add a generate flag. For this study, keeping the existing generator is enough to start — test (the quoted split) is already uniform.

### 2. Freeze + slice

```bash
bash scripts/scale_study/02_freeze_and_slice.sh
```

Writes:

- `data/scale300k/manifest_train_{6000,12000,25000,50000,120000,300000}.parquet`
- `data/scale300k/norm_stats.npz` (full train)
- `results/scale_study/frozen.json` (includes `residual_scale`)

Manifests live **next to the HDF5 shards** (same directory). That matters: generation stores `file_path` as a basename.

### 3. Train

Start small:

```bash
bash scripts/scale_study/03_train_one.sh 6k_2p5m
bash scripts/scale_study/03_train_one.sh 12k_2p5m
```

Or the whole scaling phase:

```bash
bash scripts/scale_study/03_train_grid.sh scaling
```

Capacity runs can overlap with scaling if you have GPUs:

```bash
bash scripts/scale_study/03_train_grid.sh capacity
# SLURM — scaling and capacity in parallel:
bash scripts/scale_study/slurm/submit_train_array.sh scaling
bash scripts/scale_study/slurm/submit_train_array.sh capacity
```

Checkpoints land under `logs/scale_study/<run_id>/version_*/checkpoints/`. The best file is chosen by lowest `val_mismatch` in the filename.

### 4. Eval scaling → write prediction (gate)

```bash
bash scripts/scale_study/04_eval_grid.sh scaling
# or SLURM:
bash scripts/scale_study/slurm/submit_eval_array.sh scaling

# After CSVs exist:
bash scripts/scale_study/04_fit_prediction.sh
```

This prints and saves the two headline numbers:

- predicted median mismatch at **300k**
- predicted data size for **$10^{-4}$**

Fits (with and without a floor term):

$$
m(N) = A\, N^{B}
\qquad\text{and}\qquad
m(N) = A\, N^{B} + C
$$

File: `results/scale_study/prediction.json`.

**Do not evaluate or peek at 300k until this file exists.**

### 5. Held-out 300k + capacity eval

```bash
bash scripts/scale_study/03_train_one.sh 300k_2p5m
bash scripts/scale_study/04_eval_one.sh 300k_2p5m

bash scripts/scale_study/04_eval_grid.sh capacity   # if not done yet
```

### 6. Decide

Open `scripts/scale_study/DECISION.md` and fill the table using:

```bash
python scripts/scale_study/04_eval_and_fit.py --results-dir results/scale_study
cat results/scale_study/prediction.json
```

### 7. Figure

```bash
python scripts/scale_study/05_plot_scaling.py \
  --results-dir results/scale_study \
  --out results/scale_study/fig_scale_study.png
```

- **Panel (a):** mismatch vs training waveforms, fit + 300k held-out marker  
- **Panel (b):** mismatch vs model size at 120k  

### 8. Euler-step check (branch: “something else is the floor”)

```bash
bash scripts/scale_study/06_euler_check.sh 120k_2p5m
```

Sweeps `--steps 20 50 100`. If the knee moves, set `N_STEPS` in `PROTOCOL.env` and re-eval.

---

## SLURM notes

1. Edit `PROTOCOL.env` partitions / account / gres / walltimes for your site.
2. Some sites use `--gpus=1` instead of `--gres=gpu:1` — change `SLURM_GPU_GRES` or edit the `sbatch` lines in `slurm/submit_*.sh`.
3. Array tasks are **1-indexed** (`--array=1-N`); `job_train.sh` / `job_eval.sh` read `results/scale_study/slurm_run_list.txt` (or `slurm_eval_list.txt`) line `SLURM_ARRAY_TASK_ID`.
4. Logs: `results/scale_study/slurm_logs/`.
5. Generation may need a different module/env than training. Point `PYTHON` in `PROTOCOL.env` at the env that has `gwsurrogate` for step 1, then switch `PYTHON` to the pytorch env for steps 2–7 — or set `PYTHON` per submission:

```bash
PYTHON=$HOME/micromamba/envs/igwn-py311/bin/python \
  bash scripts/scale_study/slurm/submit_generate.sh

PYTHON=$HOME/micromamba/envs/pytorch/bin/python \
  bash scripts/scale_study/slurm/submit_train_array.sh scaling
```

6. Freeze/slice (step 2) needs pytorch (for `residual_scale`) but is short — run it interactively on a login/GPU node, or wrap it in a tiny sbatch yourself.

---

## Outputs cheat-sheet

| Path | Meaning |
|------|---------|
| `data/scale300k/manifest.parquet` | Parent pool |
| `data/scale300k/manifest_train_<N>.parquet` | Nested train prefix + shared val/test |
| `data/scale300k/norm_stats.npz` | Frozen norms |
| `results/scale_study/frozen.json` | residual_scale + paths |
| `logs/scale_study/<run_id>/` | Lightning logs + checkpoints |
| `results/scale_study/eval_<run_id>.csv` | Per-waveform mismatches |
| `results/scale_study/summary.csv` | Medians / p95 / max |
| `results/scale_study/fit.json` | Power-law fits |
| `results/scale_study/prediction.json` | **Gate** before 300k |
| `results/scale_study/fig_scale_study.{png,pdf}` | Panels (a)+(b) |

---

## Tick list

- [ ] `PROTOCOL.env` created and paths set  
- [ ] 300k pool generated  
- [ ] Freeze + slice done (`frozen.json` has `residual_scale`)  
- [ ] Trained: 6k, 12k, 25k, 50k, 120k @ 2.5M  
- [ ] Trained: 120k @ 0.6M and 10M  
- [ ] Eval CSVs for the five scaling sizes  
- [ ] `prediction.json` written — numbers recorded  
- [ ] Trained + eval’d 300k @ 2.5M  
- [ ] Eval CSVs for capacity runs  
- [ ] Decision filled in `DECISION.md`  
- [ ] Figure plotted  
- [ ] (If needed) Euler check + target run  

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Unknown run_id` | Check `runs.tsv`; ids are like `6k_2p5m` |
| `Held-out gate` / missing `prediction.json` | Run `04_fit_prediction.sh` first |
| Train recomputes norms | Ensure `--data.norm_stats_path` points at frozen `norm_stats.npz` and `compute_stats_if_missing=false` (train script sets both) |
| Eval can’t find shards | Manifests must sit in `DATA_DIR` next to `waveforms_*.h5` |
| `best_checkpoint` fails | Confirm `logs/scale_study/<run_id>/version_*/checkpoints/` exists |
| OOM on 10M | Lower `--data.batch_size` via LightningCLI override in `03_train_one.sh` or the YAML |
| Generation missing `gwsurrogate` | Use an IGWN / conda env with GSL for step 1 only |

---

## Relation to the rest of the repo

- Training entrypoint: `main.py` (LightningCLI)  
- Paper eval: `eval_fm_unified.py` (not `evalute.py` defaults)  
- Post-train talk pipeline: `scripts/run_prefreeze_pipeline.sh` (orthogonal; use after you freeze `N_STEPS`)  
- This study does **not** regenerate per size — that would break nested subsets and waste CPU.
