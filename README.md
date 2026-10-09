# gw-flow-surrogate

PyTorch Lightning **decoder-only autoregressive surrogate** for quasi-circular, spin-aligned BBH gravitational waveforms. The model predicts amplitude–phase patch tokens with a **flow matching head**, conditioned on $\theta = (q, \chi_{1z}, \chi_{2z})$.

Ground truth comes from **NRHybSur3dq8** via `gwsurrogate` (geometric units, $G=c=1$).

---

## Quick map (read this first)

Steps **depend on each other**:
| Step | What | Needs from previous steps |
|------|------|---------------------------|
| **0** | Install packages | — |
| **1** | Smoke test (optional) | Step 0 only (no data) |
| **2** | Generate data | Step 0 + `gwsurrogate` / GSL |
| **3** | Set `residual_scale` in config | Step 2 (`manifest.parquet`) |
| **4** | Train | Steps 2–3 |
| **5** | Monitor training | Step 4 running (or finished logs) |
| **6** | Evaluate (paper CSV) | Step 4 checkpoint + Step 2 manifest |
| **7** | Figures (optional) | Step 6 CSVs |
| **8** | Scaling law + choose $N$ | ≥2 (ideally 3) converged runs at different train sizes, same eval protocol |

**Paper numbers** must use the frozen protocol (see Step 6). Do not mix with `evalute.py`’s default 1-patch context.

---

## 0. Required packages and install

### What you need

| Package | Role |
|---------|------|
| `torch` | Training / inference |
| `lightning` | Trainer, CLI, logging |
| `jsonargparse[signatures]` | LightningCLI configs |
| `tensorboard` | Default training monitor |
| `numpy`, `scipy` | Numerics / mismatch helpers |
| `h5py`, `pyarrow`, `pandas` | Waveform storage + manifests |
| `matplotlib` | Figures / diagnostic plots |
| `tqdm` | Progress bars |
| `gwsurrogate` | **Data generation only** (NRHybSur3dq8) |
| `pytest` | Tests (optional; not in `requirements.txt`) |
| `wandb` | Optional monitor (not required) |

Recommended: **Python ≥ 3.10**, a CUDA-capable GPU for training/eval. Data generation is mostly CPU.

### Install

```bash
cd gw-flow-surrogate

# recommended: fresh environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`gwsurrogate` often needs system GSL / LAL. If you see `libgsl.so.27: cannot open shared object file`:

```bash
conda install -c conda-forge gsl
# or use an IGWN env, e.g. igwn-py311, then pip install the rest
```

You can train/eval on **already generated** data without a working `gwsurrogate` install; you only need it for Step 2.

---

## 1. Smoke test (optional, no data)

**Needs:** Step 0 only.

```bash
python smoke_test.py
# optional unit tests:
pip install pytest
pytest tests/ -v
```

If this fails, fix the environment before generating data or training.

---

## 2. Generate data

**Needs:** Step 0 with working `gwsurrogate` (+ GSL).

**Produces:** `data/<run>/manifest.parquet`, `waveforms_*.h5`, norm stats under that directory.

Example (25k train, a good first run):

```bash
python -m data_generation.cli generate \
    --output-dir data/25k \
    --n-files 1 --n-train 25000 --n-test 500 --n-ood 500 \
    --val-fraction 0.1 --dt 0.5 --seed 42
```

Larger producation-scale set (slow / large disk):

```bash
python -m data_generation.cli generate \
    --output-dir data/120k \
    --n-files 1 --n-train 120000 --n-test 1000 --n-ood 500 \
    --val-fraction 0.1 --dt 0.5 --seed 42
```

| Split | Sampling | Purpose |
|-------|----------|---------|
| `train` | Scrambled Sobol | Training |
| `val` | Disjoint Sobol (~10%) | Validation |
| `test` | Uniform | In-domain eval |
| `ood` | Extrapolation box | OOD probe |

Tiny synthetic data (no `gwsurrogate`, for plumbing only):

```bash
python scripts/create_smoke_data.py
```

**Check:** `data/25k/manifest.parquet` exists before Step 3.

Training later uses patch-level `norm_stats.npz` written by the DataModule, not generation’s `norm_stats.json`.

---

## 3. Set `residual_scale` (required before training)

**Needs:** Step 2 manifest.

**Produces:** a float you paste into the YAML configuration (`model.residual_scale`).

```bash
python - <<'PY'
from datasets.datamodule import WaveformDataModule
from models.lightning_module import compute_residual_scale

dm = WaveformDataModule(
    manifest_path="data/25k/manifest.parquet",  # match your data
    batch_size=64,
    num_workers=0,
)
print(compute_residual_scale(dm, patch_len=32))
PY
```

Edit the matching config, e.g. `configs/stage1.yaml`:

- `model.residual_scale: <printed value>`
- `data.manifest_path: ./data/25k/manifest.parquet` (or `data/120k/...`)

Configs ship with a placeholder scale, **replace it** for the dataset you generated.

---

## 4. Train

**Needs:** Steps 2–3 (data on disk + config updated).

```bash
python main.py fit --config configs/stage1.yaml
```

Larger trunk:

```bash
python main.py fit --config configs/stage1_large.yaml
```

Overrides without editing YAML:

```bash
python main.py fit --config configs/stage1.yaml \
    --data.manifest_path data/25k/manifest.parquet \
    --data.batch_size 32 \
    --trainer.max_epochs 50
```

| Config | Purpose |
|--------|---------|
| `configs/stage1.yaml` | Default small model (~2.5M) |
| `configs/stage1_large.yaml` | Larger trunk |
| `configs/stage1_smoke.yaml` | Tiny smoke train |
| `configs/stage1_prob.yaml` | Legacy, unused by current `main.py` |

**Produces:** checkpoints under `logs/<run_name>/version_*/checkpoints/`, e.g.  
`epoch=XXX-val_mismatch=0.0YYY.ckpt`.

Monitor **`val_mismatch`** (lower is better), not `val_loss`. Pick the checkpoint with the best (lowest) `val_mismatch` for Step 6.

---

## 5. Monitor training

**Needs:** Step 4 (logger writes while training; you can also open logs afterward).

### Default: TensorBoard

Configs already use `TensorBoardLogger` → `logs/`.

In another terminal:

```bash
tensorboard --logdir logs
# open the URL it prints (usually http://localhost:6006)
```

Watch `val_mismatch` and learning rate.

### Optional: Weights & Biases

```bash
pip install wandb
wandb login
```

Run with a W&B logger override (LightningCLI):

```bash
python main.py fit --config configs/stage1.yaml \
  --trainer.logger.class_path=lightning.pytorch.loggers.WandbLogger \
  --trainer.logger.init_args.project=gw-flow-surrogate \
  --trainer.logger.init_args.name=stage1-25k
```

Use either TensorBoard **or** W&B; both are fine. The committed configs default to TensorBoard.

---

## 6. Evaluate (step by step)

**Needs:**

1. A trained checkpoint from Step 4  
2. The **same** (or compatible) `manifest.parquet` from Step 2  
3. Frozen knobs below for any number you quote in a paper/talk  

### Frozen evaluation protocol

Also stored in `results/prefreeze/PROTOCOL.txt`:

| Field | Value |
|-------|--------|
| split | `test` (first `--n-waveforms` rows) |
| `--context-fraction` | `0.5` |
| `--n-waveforms` | `500` |
| `--n-samples` / `--mean-of` | `1` / `4` |
| `--n-steps` | knee from speed frontier (typically `40`) |
| headline metric | median of `mismatch_phase_time_future` |

### 6a. Paper metrics CSV (use this for tables / scaling)

```bash
# set these to your paths
CKPT=logs/<run>/version_*/checkpoints/<best-val_mismatch>.ckpt
MANIFEST=data/25k/manifest.parquet   # or data/120k/manifest.parquet

python eval_fm_unified.py \
  --checkpoint "$CKPT" \
  --manifest "$MANIFEST" \
  --split test \
  --context-fraction 0.5 \
  --n-waveforms 500 --n-samples 1 --mean-of 4 --n-steps 40 \
  --out results/fm_eval.csv
```

**Produces:** `results/fm_eval.csv`. Report `mismatch_phase_time_future` median.

### 6b. Diagnostics / plots (optional; different default context)

Default = **1 context patch** (TODO: fix the protocol). Useful for overlays and drift plots.

```bash
python evalute.py \
  --checkpoint "$CKPT" \
  --manifest "$MANIFEST" \
  --split test \
  --n-waveforms 128 --n-samples 4 --n-steps 40 \
  --mean-of 1 --n-plot 4 \
  --output-dir eval_test
```

For paper-aligned overlays, I use `--context-fraction 0.5`.

| Script | Default context | Use for |
|--------|-----------------|---------|
| `eval_fm_unified.py` | 50% seeded | Quoted mismatches |
| `evalute.py` | 1 patch seeded | Plots / debugging |

### 6c. Representation floor (optional)

```bash
python scripts/measure_floor.py \
  --manifest "$MANIFEST" --split test --n-waveforms 200 \
  --checkpoint "$CKPT" --n-steps 40 \
  --out results/floor_metrics.json
```

### 6d. Gated full pipeline (optional)

After we are happy with a checkpoint:

```bash
bash scripts/run_prefreeze_pipeline.sh help
# edit CONFIG / env: CKPT, MANIFEST, N_STEPS, PYTHON
bash scripts/run_prefreeze_pipeline.sh phase_a
bash scripts/run_prefreeze_pipeline.sh phase_b
N_STEPS=40 bash scripts/run_prefreeze_pipeline.sh freeze_protocol
# … see script help for phase_c–f
```

Defaults inside that script may still point at local paths from a previous machine, always set `CKPT` and `MANIFEST`.

---

## 7. Figures (optional)

**Needs:** Step 6 CSVs (and sometimes the manifest). Outputs go to `paper_figs/out/`.

For the **data-scaling law** (mismatch vs training size), see **Step 8**, it needs multiple trained models, not one CSV.

```bash
python paper_figs/fig_calibration.py eval \
  --checkpoint "$CKPT" --manifest "$MANIFEST" \
  --split test --context-fraction 0.5 --n-waveforms 500 --S 8 \
  --out results/calibration.csv
python paper_figs/fig_calibration.py plot results/calibration.csv
python paper_figs/fig_speed_frontier.py eval \
  --checkpoint "$CKPT" --manifest "$MANIFEST" \
  --context-fraction 0.5 --steps 5 10 20 40 100 --n-waveforms 200 \
  --out results/speed_frontier.csv
python paper_figs/fig_speed_frontier.py plot results/speed_frontier.csv
```

---

## 8. Scaling law and estimating training data size

**Preferred path (automated):** the full nested-pool study (300k generate → freeze norms →
eight trainings → power-law gate → decision figure) lives in
[`scripts/scale_study/README.md`](scripts/scale_study/README.md).
Use that for the \(10^{-4}\) target study. The manual notes below remain as a short reference.

**Needs:** Steps 2–6 repeated at **several** training-set sizes (same model size, same frozen eval protocol). Typical points: **6k → 25k → 120k** train waveforms.

Goal: measure how median mismatch scales with $N$, then use that law to decide how much data you need for a target accuracy (e.g. $10^{-3}$).

### 8a. Produce the scaling-law figure

1. **Generate** datasets at each size (Step 2), e.g. `--n-train 6000`, `25000`, `120000`.
2. For each size: set `residual_scale` (Step 3) → **train to convergence** (Step 4) →
   keep the best `val_mismatch` checkpoint.
3. **Evaluate each checkpoint** with the **same** frozen protocol (Step 6a), ideally on the
   **same** test waveforms (same manifest / first 500 test rows):

```bash
# example: three CSVs, same MANIFEST and knobs
for label in 6k 25k 120k; do
  python eval_fm_unified.py \
    --checkpoint "logs/flow_${label}/.../<best>.ckpt" \
    --manifest data/120k/manifest.parquet \
    --split test --context-fraction 0.5 \
    --n-waveforms 500 --n-samples 1 --mean-of 4 --n-steps 40 \
    --out "results/scale_${label}.csv"
done
```

4. Read the headline median from each CSV:

```bash
python - <<'PY'
import pandas as pd
for f in ["results/scale_6k.csv", "results/scale_25k.csv", "results/scale_120k.csv"]:
    m = pd.read_csv(f)["mismatch_phase_time_future"].median()
    print(f"{f}: p50 = {m:.4e}")
PY
```

5. Paste those p50s into `paper_figs/fig_scaling_final.py` (`CONVERGED_X` / `CONVERGED_Y`),
   then plot:

```bash
# edit CONVERGED_Y = [p50_6k, p50_25k, p50_120k] in paper_figs/fig_scaling_final.py
python paper_figs/fig_scaling_final.py
```

Optional per-$q$ slopes (same CSVs):

```bash
python paper_figs/fig_scaling_by_q.py \
  --points 6000:results/scale_6k.csv \
           25000:results/scale_25k.csv \
           120000:results/scale_120k.csv \
  --out-csv results/scaling_by_q.csv
```

**Rules that keep the law honest**

- Same model capacity (e.g. small ~2.5M) across $N$.
- Same eval protocol (ctx 0.5, 500 wf, mean-of 4, same `n-steps`).
- Prefer the **same test set** for all points (otherwise sizes are not comparable).
- Use **converged** best checkpoints only; fixed-budget / unfinished runs are not fit points.

### 8b. Estimate how much training data you need

Assume a power law $\mathrm{mismatch}_{p50} \approx A\, N^{b}$ (log–log line).
With two measured points $(N_1, m_1)$, $(N_2, m_2)$:

$$
b = \frac{\log m_2 - \log m_1}{\log N_2 - \log N_1},
\qquad
A = m_1 / N_1^{b}.
$$

For a **target** mismatch $m_\star$ (e.g. $10^{-3}$):

$$
N_\star = \bigl(m_\star / A\bigr)^{1/b}
= N_1 \,(m_\star / m_1)^{1/b}.
$$

Example (fit on 6k→25k, then ask for $m_\star = 10^{-3}$):

```bash
python - <<'PY'
import math
N1, m1 = 6000, 1.1e-2
N2, m2 = 25000, 2.2e-3
m_star = 1e-3

b = (math.log(m2) - math.log(m1)) / (math.log(N2) - math.log(N1))
N_star = N1 * (m_star / m1) ** (1.0 / b)
print(f"slope b = {b:.3f}")
print(f"N needed for p50 ≈ {m_star:.0e}: ~ {N_star:.0f} train waveforms")
# check prediction at 120k:
N3 = 120000
m3_pred = m1 * (N3 / N1) ** b
print(f"predicted p50 at N={N3}: {m3_pred:.3e}  (compare to measured later)")
PY
```

**How to use this in practice**

1. Train/eval at two sizes (e.g. 6k and 25k) under the frozen protocol.  
2. Fit $b$, pick $m_\star$ (production target in the figure is $10^{-3}$).  
3. Compute $N_\star$ that is the estimated training-set size.  
4. Generate ≈ that many waveforms, train, and **measure** the new p50 (the law can bend;
   the 120k point in this project landed within ~2× of the 6k–25k extrapolation).

The dashed line in `fig_scaling_final.py` is that target $10^{-3}$, not a measured median.

---

## Repository layout

```
gw-flow-surrogate/
├── main.py                  # LightningCLI entry
├── smoke_test.py
├── eval_fm_unified.py       # paper CSV eval
├── evalute.py               # diagnostics + plots
├── configs/
├── data_generation/
├── datasets/
├── models/
├── scripts/
├── paper_figs/
├── tests/
├── results/prefreeze/PROTOCOL.txt
├── data/                    # gitignored — from Step 2
└── logs/                    # gitignored — from Step 4
```

---

## Design notes

- Absolute phase is gauge; the model predicts increments + de-ramped residuals.
- `log_a` / `Δφ`: train-split patch norm (`norm_stats.npz`); residuals ÷ `residual_scale`.
- Causal AR: position $p$ attends to $\le p$; flow targets future horizons.

---

## Citation

Neural surrogate of **NRHybSur3dq8** (Varma et al.). Ground-truth waveforms are surrogate/NR replicas within that model’s domain, not independent Einstein-equation solutions.
