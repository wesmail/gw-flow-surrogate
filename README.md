# gw-flow-surrogate

Flow-matching autoregressive surrogate for **NRHybSur3dq8** (2,2)-mode aligned-spin BBH waveforms.

## Table of contents

1. [What this is](#1-what-this-is)
2. [Installation](#2-installation)
3. [Data generation](#3-data-generation)
4. [Training](#4-training)
5. [Evaluation protocol (frozen)](#5-evaluation-protocol-frozen)
6. [Scaling study](#6-scaling-study)
7. [Speed](#7-speed)
8. [Parameter estimation with ET (2L)](#8-parameter-estimation-with-et-2l)
9. [Repository layout](#9-repository-layout)
10. [Reproducibility notes and known limitations](#10-reproducibility-notes-and-known-limitations)
11. [Citation / acknowledgements](#11-citation--acknowledgements)

---

## 1. What this is

This repository trains and evaluates a neural surrogate for the (2,2) mode of
quasi-circular, spin-aligned binary black hole waveforms from **NRHybSur3dq8**.
The intrinsic parameters are $\theta = (q,\chi_{1z},\chi_{2z})$. Each waveform
on a fixed geometric-time grid is turned into amplitude and phase tracks, cut
into patches, and modelled by a decoder-only transformer: given earlier patches
(and $\theta$), a **flow-matching** head draws the next patch. At inference the
first patch is the true seed (16 M of signal); the rest is generated
autoregressively.

**Pipeline at a glance**

1. Generate a waveform pool (Sobol train/val, uniform test, OOD) with `gwsurrogate`.
2. Freeze patch normalisation and `residual_scale`, then train.
3. Evaluate with the frozen protocol (12 Euler steps, `mean_of=1`).
4. Run the nested data-scaling study (six sizes × seeds).
5. Time the surrogate and compare to NRHybSur3dq8.
6. Run parameter estimation on an ET 2L network (reference vs our model).

---

## 2. Installation

Two environments are used.

| Env | Role |
|-----|------|
| `pytorch` | Training, evaluation, paper figures (CUDA PyTorch + Lightning). |
| `igwn-py311` | Data generation and PE: `gwsurrogate`, LALSuite, bilby, nessai. For GPU PE this env also needs a CUDA-capable PyTorch. |

From the repo root:

```bash
# training / eval environment (example)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# plus a CUDA build of torch matching your driver

# PE / generation: use your IGWN-style env with gwsurrogate, bilby, lalsimulation, nessai
```

Check that the GPU is visible in the env you will use for training or PE:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**External data (not in git)**

- **NRHybSur3dq8 (gwsurrogate).** Downloaded once:

  ```bash
  python scripts/gen_shard.py download
  ```

  The file lives under gwsurrogate’s download directory. Loading the surrogate
  does **not** re-download if the `.h5` is already present.

- **LAL waveform data** (for wrapper check A against bilby’s LAL
  `NRHybSur3dq8`). Zenodo record
  [14999310](https://zenodo.org/records/14999310): file
  `NRHybSur3dq8_lal_v1.0.h5` (MD5 `5ffba6443e3508f0f79f70b5d64eaf60`). Keep it
  **outside** the repo and point LAL at it:

  ```bash
  export LAL_DATA_PATH=/path/to/lal_data_dir   # directory that contains the .h5
  ```

- **ET noise curves.** Official CoBA HFLF-cryo ASDs; download commands and
  SHA-256 sums are in [`asd/README.md`](asd/README.md). Expected files:

  ```text
  asd/ET_HFLF_cryo_15km.txt   # 2L, 15 km
  asd/ET_HFLF_cryo_10km.txt   # triangle, 10 km
  ```

---

## 3. Data generation

Waveforms are generated with NRHybSur3dq8 on a fixed grid
($t/M \in [-5000,\,130]$, $\mathrm{d}t = 0.5\,\mathrm{M}$, modes `(2,2)`).
Train and validation use a scrambled Sobol sequence in the box
$q\in[1,8]$, $\chi_{1z},\chi_{2z}\in[-0.8,0.8]$; test is uniform in-box; OOD
is outside the box. Outputs are HDF5 shards plus a Parquet manifest.

The flagship pool (`data/n1000k/`) has **1 000 000** train, **100 000** val
(`val_fraction=0.1` of train), **5000** test, **500** OOD, in **64** shards
(~180 GB). Seed **42**.

### Laptop (one process)

```bash
python scripts/gen_shard.py download

python -m data_generation.cli generate \
  --output-dir data/n1000k \
  --n-train 1000000 --n-test 5000 --n-ood 500 \
  --n-files 64 --val-fraction 0.1 --seed 42
```

This writes shards, combines `data/n1000k/manifest.parquet`, and also writes
generation-level `norm_stats.json` (not the training `.npz`). Rough runtime:
many CPU-days for the full 1.1 M pool; use the cluster path below for production.

### Cluster (recommended)

Edit `cluster/env.sh` (modules / conda) and the `#SBATCH --partition` /
`--account` lines in the `.sbatch` files. Place small frozen references under
`data/ref/` (a laptop `manifest.parquet` and the frozen `norm_stats.npz`):

```bash
mkdir -p data/ref
# copy an existing audited pool's manifest + norm_stats.npz into data/ref/
#   data/ref/n1000k_manifest.parquet
#   data/ref/norm_stats.npz
```

```bash
python scripts/gen_shard.py plan \
  --infer-from data/ref/n1000k_manifest.parquet \
  --reference  data/ref/n1000k_manifest.parquet
# expect: PLAN MATCHES THE REFERENCE

sbatch --array=0-63 cluster/gen_pool.sbatch
# wait; re-submit is safe (finished shards are skipped)

bash cluster/finalize_pool.sh
```

`finalize_pool.sh` combines shard manifests, compares to the reference,
**copies** the frozen `norm_stats.npz` into the pool (never recomputes), and
builds nested training manifests. Details: [`cluster/README.md`](cluster/README.md).

### Normalisation and `residual_scale`

Training uses **patch-level** `norm_stats.npz` (keys `log_a_mean/std`,
`delta_phi_mean/std`), computed from the train split via
`datasets.waveform_dataset.compute_norm_stats` (or by the Lightning DataModule
when `compute_stats_if_missing=true`). For the paper pool this file is
**frozen** once and reused for every scaling size.

There is **no** `scripts/make_norm_stats.py` and **no** `residual_scale.txt`.
The phase-residual scale is a scalar in the training config
(`model.residual_scale: 0.08152` in `configs/n1000k_2p5m_final.yaml`), obtained
once with `models.lightning_module.compute_residual_scale` on the pool.

Example (after a pool exists):

```bash
python - <<'PY'
from datasets.datamodule import WaveformDataModule
from models.lightning_module import compute_residual_scale
dm = WaveformDataModule(
    manifest_path="data/n1000k/manifest.parquet",
    norm_stats_path="data/n1000k/norm_stats.npz",
    batch_size=128, patch_len=32, context_patches=1,
    compute_stats_if_missing=False,
)
print(compute_residual_scale(dm, patch_len=32))
PY
```

### Nested subsets (scaling study)

```bash
python scripts/make_subset_manifests.py \
  --manifest data/n1000k/manifest.parquet \
  --sizes 16000 40000 100000 251000 631000 1000000
```

Writes `data/n1000k/manifest_train_<N>.parquet` (shared val/test/ood; first $N$
train rows). Waveform files are not copied.

---

## 4. Training

The flagship recipe is `configs/n1000k_2p5m_final.yaml`: ~3.1 M parameters
(paper label “2.5 M”), batch 128, bf16-mixed, cosine LR with 500-step warmup,
`dphi_weight=30`, `la_weight=10`, `residual_scale=0.08152`.

**Schedule.** `max_steps = 320333 = 41 × 7813`. With the scaling overrides,
validation and checkpoints fire every **7813** steps so the last file is exactly
the end of the schedule.

**The model used in the paper** is the end-of-schedule checkpoint
`step=320333.ckpt` (not early stopping by `val_mismatch`). When training with
the flagship config alone, Lightning also writes `last.ckpt` at the same step
count; for PE and figures set:

```bash
export FT=logs/scaling/N1000000/seed0/checkpoints/step=320333.ckpt
# or your flagship last.ckpt after 320333 steps
```

Flagship fit (full 1 M train manifest):

```bash
python main.py fit --config configs/n1000k_2p5m_final.yaml
```

Resume (Lightning):

```bash
python main.py fit --config configs/n1000k_2p5m_final.yaml \
  --ckpt_path path/to/last.ckpt
```

Logs go under `logs/` (TensorBoard). Rough runtime: order **one GPU-day** per
full 320 k-step run on a modern NVIDIA GPU (depends on disk and batch I/O).

---

## 5. Evaluation protocol (frozen)

Canonical file: [`results/protocol/PROTOCOL.yaml`](results/protocol/PROTOCOL.yaml)
(`n_steps: 12`, `mean_of: 1`).

| Knob | Value |
|------|--------|
| Generation | $\theta$-only (context fraction 0 → one seed patch) |
| Start token | true first patch (32 samples = 16 M at $\mathrm{d}t=0.5\,\mathrm{M}$) |
| Flow steps | 12 Euler steps |
| Samples | `mean_of=1`, `n_samples=1`, `temperature=1`, seed `1234` |
| Amplitude | PCHIP |
| Metric | full-waveform white-noise mismatch, optimised over time and phase (`mm_pt_full`) |
| Test set | 5000 waveforms from `data/n1000k/manifest.parquet` split `test` |
| Tuning / scaling | rows **0–999** |
| Headline | rows **1000–4999**, with bootstrap confidence intervals in the figure scripts |
| Checkpoint | final step of the 320 333-step schedule |

Individual waveforms can differ slightly between CPU and GPU (fp rounding over
~320 autoregressive steps). Statistics over many waveforms are stable; always
compare on the **same** hardware.

### Headline evaluation

```bash
python paper_figs/fig_context_fraction.py eval \
  --checkpoint "$FT" \
  --manifest data/n1000k/manifest.parquet \
  --norm-stats data/n1000k/norm_stats.npz \
  --split test --n-waveforms 5000 --fractions 0 \
  --n-steps 12 --mean-of 1 --seed 1234 --amp-mode pchip \
  --out results/headline_K12_M1.csv
```

Produces `results/headline_K12_M1.csv` and a `.json` sidecar (settings used by
downstream figures). GPU; tens of minutes for 5000 waveforms.

```bash
python paper_figs/fig_context_fraction.py plot results/headline_K12_M1.csv
```

### Paper figure scripts

| Script | What it shows | Command | Outputs |
|--------|---------------|---------|---------|
| `fig_context_fraction.py` | Accuracy vs true seed length; $\theta$-only is the headline | `eval` / `plot` as above; for a seed sweep add e.g. `--fractions 0 0.25 0.5 0.75` | CSV + `paper_figs/out/` figures |
| `fig_param_map.py` | Error map in $(q,\chi_{\mathrm{eff}})$ | `python paper_figs/fig_param_map.py results/headline_K12_M1.csv --rows headline --out fm_param_map` | `paper_figs/out/fm_param_map.*` |
| `fig_waveform.py` | True vs model waveforms (median / hard cases) | `python paper_figs/fig_waveform.py --checkpoint "$FT" --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz --eval-csv results/headline_K12_M1.csv --out fm_waveform` | `paper_figs/out/fm_waveform.*` |
| `fig_dephasing.py` | Phase / amplitude error vs time | `python paper_figs/fig_dephasing.py --checkpoint "$FT" --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz --eval-csv results/headline_K12_M1.csv --out fm_dephasing` | `paper_figs/out/fm_dephasing.*` |
| `fig_calibration.py` | Does sample spread predict true error? | `python paper_figs/fig_calibration.py eval --checkpoint "$FT" --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz --eval-csv results/headline_K12_M1.csv --S 8 --n-waveforms 1000 --out results/calibration_K12.csv` then `python paper_figs/fig_calibration.py plot results/calibration_K12.csv` | CSV + figures (~10 min GPU for eval) |
| `fig_scaling_final.py` | Data-scaling law from summary CSV | `python paper_figs/fig_scaling_final.py --summary results/paper/scaling_summary.csv` | `paper_figs/out/` |
| `fig_speed_frontier.py` | Timing and speed–accuracy frontier | see [§7](#7-speed) | CSV / JSON + figures |
| `fig_scaling_by_q.py` | Per-$q$ slopes (older multi-CSV diagnostic) | `python paper_figs/fig_scaling_by_q.py --points N:csv ...` | `paper_figs/out/` |
| `fig_q_anatomy.py` | $q$-inversion diagnostic (older pair of CSVs) | `python paper_figs/fig_q_anatomy.py --csv-25k ... --csv-120k ... --manifest ...` | figures |

---

## 6. Scaling study

One pool; nested Sobol prefixes $N \in \{16\mathrm{k},\,40\mathrm{k},\,100\mathrm{k},\,251\mathrm{k},\,631\mathrm{k},\,1\mathrm{M}\}$;
same val/test; **same** frozen `norm_stats.npz` and `residual_scale`; identical
architecture and **320 333** steps for every run. Seeds typically `0`, then
`1 2`.

Overrides: [`configs/scaling_overrides.yaml`](configs/scaling_overrides.yaml)
(checkpoint every 7813 steps → `step=320333.ckpt`).

```bash
# after finalize_pool.sh (or make_subset_manifests.py)
DRY=1 bash cluster/submit_scaling.sh          # print plan
bash cluster/submit_scaling.sh                # SIZES default, SEED=0
SEEDS="1 2" bash cluster/submit_scaling.sh    # more seeds
SIZES="100000 1000000" SEEDS="0" CHAIN=2 bash cluster/submit_scaling.sh
```

Each job (`cluster/train_scaling.sbatch`) trains (or resumes from `last.ckpt`)
then evaluates rows 0–999 with the frozen protocol into
`results/scaling/N<N>_seed<S>.csv`.

Summarise and plot:

```bash
python scripts/make_scaling_summary.py \
  --glob 'results/scaling/N*_seed*.csv' \
  --out results/paper/scaling_summary.csv

python paper_figs/fig_scaling_final.py \
  --summary results/paper/scaling_summary.csv
```

---

## 7. Speed

Generation is sequential: **320** patches × (1 + `n_steps`) network passes per
waveform. At batch 1024 on a laptop GPU the frozen path is about **6 ms per
waveform**, comparable to NRHybSur3dq8 on one CPU core for the same grid.
`generate_fast` / `torch.compile` exists for PE and figures (`--fast` on
`fig_context_fraction.py eval`) but is **not** a large-batch speed-up.

```bash
python paper_figs/fig_speed_frontier.py time --checkpoint "$FT" \
  --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
  --steps 6 8 10 12 15 20 40 80 --batch-sizes 1 16 64 256 1024 \
  --out results/speed_model.csv

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python paper_figs/fig_speed_frontier.py ref --manifest data/n1000k/manifest.parquet \
  --n 50 --out results/speed_nrhybsur.json

python paper_figs/fig_speed_frontier.py plot results/speed_model.csv \
  --ref results/speed_nrhybsur.json \
  --acc-glob "results/sampler*/K*_M1.csv" --frozen-steps 12
```

Optional warm $\theta\to h_{22}$ timing as used in PE:

```bash
python pe/time_fm.py --checkpoint "$FT" \
  --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
  --batches 64 256 1024
```

---

## 8. Parameter estimation with ET (2L)

Use the **`igwn-py311`** env (gwsurrogate + bilby + nessai + CUDA torch).

### Design

- Zero-noise injection of **NRHybSur3dq8** (detector-frame $m_1=200\,M_\odot$,
  $m_2=100\,M_\odot$, $\chi=0.3/-0.2$) through the same wrapper as recovery.
- Network: ET **2L** = Sardinia + Meuse-Rhine, 15 km arms, relative orientation
  45°, CoBA HFLF-cryo ASD (`asd/ET_HFLF_cryo_15km.txt`). Detector helpers:
  `pe/et_detectors.py` (names `ETS`, `ETMR`).
- Sky fixed (defaults place both detectors near median antenna response);
  orbital phase marginalised analytically.
- Eight sampled parameters: chirp mass, mass ratio, $\chi_1$, $\chi_2$,
  luminosity distance, $\theta_{jn}$, $\psi$, geocent time.
- Sampler: **nessai**, vectorised GPU likelihood (fixed batch, padded).
- Grid: SNR $\in\{15,30,60,120\}$ × {NRHybSur3dq8 reference, our model}.

### Step 1 — wrapper checks

```bash
export LAL_DATA_PATH=/path/to/lal_data_dir
export FT=path/to/step=320333.ckpt

python pe/check_wrapper.py --checkpoint "$FT" \
  --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
  --asd asd/ET_HFLF_cryo_15km.txt --out results/pe_check.json
```

| Check | Meaning | Good values |
|-------|---------|-------------|
| A | wrapper(NR) vs LAL NRHybSur3dq8, ET-weighted mismatch | $\sim 0$ (need LAL data; else `--skip-lal`) |
| B | wrapper(model) vs wrapper(NR); $\rho^\star = 1/\sqrt{2\,\mathrm{mm}}$ | mm small; $\rho^\star$ tens |
| C | same $\theta$ alone vs in a batch | max relative diff $\ll 10^{-5}$ |
| D | timing for a batch | printed seconds / ms per waveform |

### Step 2 — likelihood / timing smoke test

```bash
python pe/run_pe.py --model nr --snr 30 --test --checkpoint "$FT" \
  --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
  --asd asd/ET_HFLF_cryo_15km.txt

python pe/run_pe.py --model fm --snr 30 --test --checkpoint "$FT" \
  --manifest data/n1000k/manifest.parquet --norm-stats data/n1000k/norm_stats.npz \
  --asd asd/ET_HFLF_cryo_15km.txt
```

Prints network SNR, the total-mass prior cut, **ln L at the truth** (for the
reference model, max-over-phase ln L equals $\rho^2/2$), and ms per likelihood.
Writes `results/pe/<model>_snr<snr>/test.json`. About 1–2 minutes including
compile on a 4090-class GPU.

### Step 3 — all eight runs

```bash
export FT=path/to/step=320333.ckpt
export LAL_DATA_PATH=/path/to/lal_data_dir
bash pe/run_all_pe.sh
# ONLY=fm   or ONLY=nr
# SNRS="30 60"
# OUT=results/pe
```

Reference (`nr`) jobs start in parallel with `nohup`; FM jobs run sequentially
on the GPU. Logs: `results/pe/logs/`. nessai **resumes** if interrupted.
Per-SNR wall time is hours (see the `--test` timing line × ~$10^5$–$10^6$
likelihood calls).

Outputs per run in `results/pe/<model>_snr<snr>/`:

- `posterior.csv` (including derived $q$, $M$, $\chi_{\mathrm{eff}}$)
- `summary.json` (truth, SNR, medians, 90 % intervals, evidence, timing)
- nessai state / plots

### Step 4 — PE figures and table

`pe/plot_pe.py` — **coming next** (not in the repository yet).

---

## 9. Repository layout

```text
asd/                  ET ASD files (download; see asd/README.md) + README
cluster/              SLURM helpers (env.sh, gen_pool, finalize, train/submit scaling)
configs/              Flagship + scaling YAML
data_generation/      NRHybSur3dq8 evaluation, sharding, sampling
datasets/             Patch dataset + DataModule + norm_stats.npz helper
models/               Transformer, flow head, Lightning module
paper_figs/           Paper figure scripts + style helpers
pe/                   PE wrappers, ET 2L detectors, checks, run_all_pe.sh
scripts/              gen_shard, subset manifests, scaling summary, utilities
tests/                pytest suite
results/protocol/     Frozen PROTOCOL.yaml and audit notes
main.py               Lightning CLI entry (fit / validate / test)
requirements.txt      Core Python deps for training/eval
```

---

## 10. Reproducibility notes and known limitations

- The training window is **5000 M** before merger. For a ~5 Hz start that
  implies **heavy detector-frame** binaries (roughly $\gtrsim 200$–$300\,M_\odot$
  total mass, depending on mass ratio).
- Inference is $\theta$-only but still needs the **true first patch** as the
  start token (same convention as training/evaluation).
- Accuracy is worst near the edge of the training box ($q \gtrsim 7$,
  $\chi_{\mathrm{eff}} \gtrsim 0.6$); see the parameter map.
- The generative spread (calibration figure) captures only **part** of the
  error; a shared bias remains.
- CPU vs GPU: per-waveform samples can differ slightly; aggregate metrics
  should be compared on one device.

---

## 11. Citation / acknowledgements

- **NRHybSur3dq8:** Varma et al., Phys. Rev. D 99, 064045 (2019).
- **gwsurrogate**, **bilby**, **nessai**, **LALSuite** developers and
  maintainers.
- **ET detectors and CoBA noise curves:** Branchesi et al. 2023,
  [arXiv:2303.15923](https://arxiv.org/abs/2303.15923).

Paper citation: _TBD_.
