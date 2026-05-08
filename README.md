# PINNs for Particle Trajectory Prediction in the CMS Silicon Pixel Detector

**Physics-Informed Learning of Track Direction Parameters from Pixel Doublets**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Data: CERN Open Data](https://img.shields.io/badge/data-CERN%20Open%20Data-red.svg)](https://opendata.cern.ch)

> Atınç Baş, Altan Çakır  
> Department of Physics Engineering & Department of Data Science and Analytics  
> Istanbul Technical University

---

## Overview

This repository contains the full implementation for the paper *"PINNs for Particle Trajectory Prediction in the CMS Silicon Pixel Detector"*. We develop and benchmark a **Physics-Informed Neural Network (PINN)** against a purely supervised baseline (Pure NN) for predicting charged-particle track direction parameters — azimuthal angle **φ** and pseudorapidity **η** — from hit doublets in the CMS Silicon Pixel Detector.

The helical equations of motion under a solenoidal magnetic field are embedded into the network's loss function as a differentiable ODE residual via PyTorch autograd. To address training instabilities common in PINN literature (gradient conflicts, mode collapse, loss scaling), we introduce:

- **EMA-based physics-loss normalization** to prevent scale imbalance
- **Kinematic-regime-aware soft-guided weighting** (pT, ΔR, charge)
- **Warm-up + ramp scheduling** for the physics loss weight λ_pde

---

## Key Results

All metrics evaluated on the held-out test set (N = 807,160, 15% of full dataset).

| Model   | MSE_φ | MAE_φ | R²_φ  | MSE_η | MAE_η | R²_η  |
|---------|-------|-------|-------|-------|-------|-------|
| PINN    | 1.874 | 0.990 | 0.359 | 0.901 | 0.783 | 0.597 |
| Pure NN | **1.190** | **0.757** | **0.593** | **0.733** | **0.617** | **0.672** |

### Selected Findings

- **Systematic training pathology, not kinematic instability.** The R²_φ gap between PINN and Pure NN is constant across all pT bins (ΔR²_φ ≈ 0.23–0.24), ruling out low-pT helical approximation breakdown as the failure mode.
- **Barrel η estimation is geometrically constrained.** For |η| < 1 (barrel, N = 206,792), both models achieve R²_η < −1.6, worse than a constant predictor. The longitudinal doublet displacement in the barrel (σ_Δz = 11.4 cm) provides insufficient z-leverage for η estimation; endcap tracks have σ_Δz = 20.5 cm and R²_η ≈ 0.56–0.85.
- **Physics constraint acts as a soft regularizer.** The raw (unnormalized) L_pde stabilizes at ≈ 1.13 (dimensionless, ω_scale ≈ 0.089 rad/cm), indicating the ODE is not tightly satisfied at convergence.
- **Periodic output parameterization is needed.** 15.25% of PINN φ predictions and 12.27% of Pure NN predictions fall outside [−π, π]. Modular wrap correction degrades R²_φ from +0.36 to −0.56, confirming these are genuine prediction errors.
- **EMA normalization eliminates mode collapse.** Previous runs without stabilization showed φ predictions collapsing to a constant. The proposed stabilization patches resolve this.

---

## Architecture

Both models use the same backbone:

```
Input x_norm (12 features)
    ↓
Dense(64, tanh)  →  Dense(64, tanh)  →  Dense(64, tanh)
    ↓
Linear(2)  →  [φ̂_norm, η̂_norm]
    ↓
De-standardize  →  [φ̂, η̂]
```

The PINN adds an ODE residual path: at collocation points τ ∈ [0, 1] along the doublet segment, the network's outputs are differentiated with respect to the normalized radial coordinate u via autograd, and the residuals of the helical equations of motion are penalized.

Total loss: **L = L_data + λ_pde · L̃_pde**

---

## Dataset

We use CMS Open Data–derived pixel track doublets from the CERN Open Data Portal.

| Property | Value |
|----------|-------|
| Source | CMS TTbar 13 TeV, PU = 50 |
| Total samples | 5,381,062 |
| Train / Val / Test | 3,768,895 / 805,007 / 807,160 (70 / 15 / 15 %) |
| Raw file size | ~627 MB (622 columns) |
| Features used | 12 (hit coordinates, cylindrical radii, pT, PU, BunchCrossing, charge) |
| Targets | φ (outTpPhi), η (outTpEta) |

**Download the raw data file (~627 MB):**

```bash
wget "http://opendata.cern.ch/eos/opendata/cms/datascience/CNNPixelSeedsProducerTool/TTbar_13TeV_PU50_PixelSeeds/TTbar_PU50_pixelTracksDoublets_0_final.h5"
```

> The HDF5 file uses Pandas fixed format. Due to a Python 3.10+ compatibility issue with `pd.read_hdf`, the training script reads it directly with `tables` (PyTables). No preprocessing step needed — `cms_pinn.py` handles this automatically.

**Demo data (no download needed):**  
`df_test_demo.parquet` — a 1.3 MB synthetic demo test set included in this repository, suitable for testing `analysis.py` locally without the full CERN dataset.

---

## Installation

```bash
git clone https://github.com/AtincBas/pinn-cms-project.git
cd pinn-cms-project
pip install -r requirements.txt
```

> **Training requires a GPU.** We recommend [Google Colab](https://colab.research.google.com/) (free tier is sufficient for 100 epochs). Analysis runs on CPU.

---

## Usage

### Step 1 — Train (Google Colab / GPU)

Upload `cms_pinn.py` and the downloaded `.h5` data file to Colab, then:

```bash
python cms_pinn.py \
    --epochs_pinn 100 \
    --epochs_pure 100 \
    --batch_size  512 \
    --lambda_pde  0.02 \
    --output_dir  .
```

This produces:
- `df_test.parquet` — test set predictions (features + φ̂/η̂ for both models)
- `history_pinn.csv` — per-epoch training history (includes raw L_pde and EMA values)
- `history_pure.csv` — Pure NN training history

Download `df_test.parquet` from Colab to run local analysis.

### Step 2 — Analyse (local, CPU)

```bash
python analysis.py \
    --df_test  df_test.parquet \
    --steps    1,2,3,4,5,6,7,8 \
    --out_dir  figures \
    --n_total  5381062 \
    --n_train  3768895 \
    --n_val    805007
```

**Quick demo (no Colab needed):**

```bash
python analysis.py --df_test df_test_demo.parquet
```

### Analysis Steps

| Step | Description |
|------|-------------|
| 1 | Dataset overview: sizes, sentinel fraction (inTpPt = −1.0), φ range violations |
| 2 | pT sentinel split: compare real-pT vs sentinel performance |
| 3 | Kinematic binning: R² by pT, \|η\|, ΔR, and charge bins |
| 4 | Bias scatter plots: φ error vs φ_true and vs pT |
| 5 | Physics residual: wrap correction effect on φ metrics |
| 6 | Summary metrics table: MSE, MAE, R² for both models |
| 7 | Normalized vs raw L_pde: EMA reconstruction from training history |
| 8 | Barrel analysis (\|η\| < 1): η distribution comparison, Δz statistics |

---

## Repository Structure

```
pinn-cms-project/
├── cms_pinn.py              # Training pipeline (data loading → model → export)
├── analysis.py              # 8-step analysis pipeline (runs on df_test.parquet)
├── CMS_PINN.ipynb           # Original Colab notebook (reference implementation)
├── df_test_demo.parquet     # Synthetic demo test set (1.3 MB, for local testing)
├── figures/                 # Generated plots (bias scatter, barrel η distribution)
│   ├── step4a_phi_err_vs_phi_true.png
│   ├── step4b_phi_err_vs_pt.png
│   └── step8_eta_dist_barrel.png
├── CROSS_VALIDATION_REPORT.md  # Numerical verification of all paper tables
├── requirements.txt
├── LICENSE
└── README.md
```

> Large files not tracked by git: `*.h5`, `df_test_real.parquet`, `df_test_realistic.parquet`, `TTbar_*.parquet`. See `.gitignore`.

---

## Reproducibility Notes

- **Random seed:** All experiments use `random_state=42` (sklearn splits) and `torch.manual_seed(42)`.
- **Data split:** Stratified by none; two-stage `train_test_split` (test_size=0.15, then test_size=0.176) yielding 70/15/15.
- **Normalization:** z-score using training-set statistics only; applied independently to features and targets.
- **Hardware:** Training performed on Google Colab (NVIDIA T4/A100). Analysis runs on CPU.

---

## Citation

If you use this code or dataset in your work, please cite:

```bibtex
@article{cakir2026pinncms,
  title   = {PINNs for Particle Trajectory Prediction in the CMS Silicon Pixel Detector},
  author  = {{\c{C}}ak{\i}r, Altan and Ba{\c{s}}, At{\i}n{\c{c}}},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://github.com/AtincBas/pinn-cms-project}
}
```

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
2. Krishnapriyan, A. et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*, 34.
3. Moseley, B., Markham, A., & Nissen-Meyer, T. (2023). Finite basis physics-informed neural networks (FBPINNs). *Advances in Computational Mathematics*, 49, 62.
4. CERN Open Data Portal. https://opendata.cern.ch (accessed January 2026).

---

## License

MIT — see [LICENSE](LICENSE).
