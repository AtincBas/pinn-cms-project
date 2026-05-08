# Physics-Informed Neural Networks for Particle Tracking in the CMS Silicon Pixel Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Data: CERN Open Data](https://img.shields.io/badge/data-CERN%20Open%20Data-red.svg)](https://opendata.cern.ch)

> Atınç Baş, Altan Çakır  
> Istanbul Technical University

---

## What This Is

A PINN that predicts charged-particle track parameters — azimuthal angle **φ** and pseudorapidity **η** — from hit doublets in the CMS Silicon Pixel Detector. The helical equations of motion under a 3.81 T solenoidal field are embedded directly into the loss function as a differentiable ODE residual.

Trained against a pure MLP baseline (same architecture, no physics constraint) on CMS Open Data (TTbar 13 TeV, PU=50).

---

## Results

Test set: N = 807,160 (15% of full dataset).

| Model   | MSE_φ | MAE_φ | R²_φ  | MSE_η | MAE_η | R²_η  |
|---------|-------|-------|-------|-------|-------|-------|
| PINN    | 1.874 | 0.990 | 0.359 | 0.901 | 0.783 | 0.597 |
| Pure NN | **1.190** | **0.757** | **0.593** | **0.733** | **0.617** | **0.672** |

The R²_φ gap is constant across all pT bins (ΔR²_φ ≈ 0.23–0.24), pointing to a systematic training issue rather than kinematic instability. For barrel tracks (|η| < 1), both models score R²_η < −1.6 — the longitudinal doublet spacing in the barrel doesn't carry enough z-information for η estimation.

---

## Architecture

Both models share the same backbone:

```
Input (12 features, z-score normalized)
    ↓
Dense(64, tanh) → Dense(64, tanh) → Dense(64, tanh)
    ↓
Linear(2) → [φ̂_norm, η̂_norm]
    ↓
De-standardize → [φ̂, η̂]
```

The PINN adds an ODE residual path: at random collocation points along the doublet segment, network outputs are differentiated with respect to the normalized radial coordinate via autograd, and the helical residuals are penalized. Training uses EMA-based L_pde normalization to prevent loss scale imbalance and a 10-epoch warmup + 20-epoch ramp on λ_pde.

---

## Dataset

CMS pixel track doublets from the CERN Open Data Portal.

| Property | Value |
|----------|-------|
| Source | CMS TTbar 13 TeV, PU = 50 |
| Total samples | 5,381,062 |
| Train / Val / Test | 3,768,895 / 805,007 / 807,160 |
| Features | 12 (hit coordinates, cylindrical radii, pT, PU, BunchCrossing, charge) |
| Targets | φ (outTpPhi), η (outTpEta) |

**Download the raw data file (~627 MB):**

```bash
wget "http://opendata.cern.ch/eos/opendata/cms/datascience/CNNPixelSeedsProducerTool/TTbar_13TeV_PU50_PixelSeeds/TTbar_PU50_pixelTracksDoublets_0_final.h5"
```

The HDF5 file uses Pandas fixed format. Due to a Python 3.10+ incompatibility with `pd.read_hdf`, the training script reads it via `tables` (PyTables) directly — no preprocessing needed.

**Demo data:** `df_test_demo.parquet` (1.3 MB, included) lets you run `analysis.py` locally without downloading the full dataset.

---

## Installation

```bash
git clone https://github.com/AtincBas/pinn-cms-project.git
cd pinn-cms-project
pip install -r requirements.txt
```

Training requires a GPU. [Google Colab](https://colab.research.google.com/) free tier works fine for 100 epochs.

---

## Usage

### Train

Upload `cms_pinn.py` and the `.h5` data file to Colab, then:

```bash
python cms_pinn.py \
    --epochs_pinn 100 \
    --epochs_pure 100 \
    --batch_size  512 \
    --lambda_pde  0.02 \
    --output_dir  .
```

Outputs:
- `df_test.parquet` — test set with predictions for both models
- `history_pinn.csv` — per-epoch loss breakdown (data, PDE raw, PDE normalized, EMA)
- `history_pure.csv` — Pure NN training history

Download `df_test.parquet` from Colab to run analysis locally.

### Analyse

```bash
python analysis.py \
    --df_test  df_test.parquet \
    --steps    1,2,3,4,5,6,7,8 \
    --out_dir  figures
```

**Quick demo (no Colab needed):**

```bash
python analysis.py --df_test df_test_demo.parquet
```

### Analysis Steps

| Step | What it shows |
|------|--------------|
| 1 | Dataset sizes, sentinel fraction (inTpPt = −1.0), φ range violations |
| 2 | Real-pT vs sentinel-pT performance split |
| 3 | R² binned by pT, \|η\|, ΔR, charge |
| 4 | Bias scatter: φ error vs φ_true and vs pT |
| 5 | Wrap correction effect on φ metrics |
| 6 | Summary metrics table (MSE, MAE, R²) |
| 7 | Raw L_pde and EMA reconstruction from training history |
| 8 | Barrel analysis: |η| < 1 η distribution, Δz statistics |

---

## Repository

```
pinn-cms-project/
├── cms_pinn.py          # Training script
├── analysis.py          # Analysis pipeline
├── CMS_PINN.ipynb       # Original Colab notebook
├── df_test_demo.parquet # Demo test set for local analysis
├── figures/             # Sample output plots
├── requirements.txt
├── LICENSE
└── README.md
```

Not tracked by git: `*.h5`, `df_test.parquet`, `history_*.csv`, model checkpoints. See `.gitignore`.

---

## Reproducibility

- Random seed: `random_state=42` (sklearn), `torch.manual_seed(42)`
- Data split: two-stage `train_test_split` (test_size=0.15, then test_size=0.176) → 70/15/15
- Normalization: z-score from training set only
- Hardware: Google Colab (NVIDIA T4/A100)

---

## License

MIT — see [LICENSE](LICENSE).
