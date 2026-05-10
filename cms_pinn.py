"""
cms_pinn.py — CMS Pixel Doublets: PINN vs Pure NN Training Pipeline

Usage (Google Colab / GPU):
    python cms_pinn.py --epochs_pinn 100 --epochs_pure 100

Outputs (saved to --output_dir):
    df_test.parquet      — test-set predictions for both models
    history_pinn.csv     — per-epoch PINN training history (includes raw L_pde and EMA)
    history_pure.csv     — per-epoch Pure NN training history

Data:
    Expects TTbar_PU50_pixelTracksDoublets_0_final.h5 in the current directory.
    Download: http://opendata.cern.ch/eos/opendata/cms/datascience/
              CNNPixelSeedsProducerTool/TTbar_13TeV_PU50_PixelSeeds/
              TTbar_PU50_pixelTracksDoublets_0_final.h5

Reference:
    Baş, A. & Çakır, A. (2026). PINNs for Particle Trajectory Prediction
    in the CMS Silicon Pixel Detector. Istanbul Technical University.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CMS PINN vs Pure NN training pipeline")
    p.add_argument("--max_samples",  type=int,   default=None,
                   help="Subsample N rows from full dataset (None = all ~5.38M)")
    p.add_argument("--epochs_pinn",  type=int,   default=100)
    p.add_argument("--epochs_pure",  type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--lambda_pde",   type=float, default=0.02)
    p.add_argument("--batch_size",   type=int,   default=512)
    p.add_argument("--pde_frac",     type=float, default=1.0,
                   help="Fraction of each batch used for PDE residual (0.25 = 4x faster, "
                        "use 1.0 for paper-exact reproducibility)")
    p.add_argument("--output_dir",   default=".",
                   help="Directory for df_test.parquet and history CSVs")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "inX", "inY", "inZ",
    "outX", "outY", "outZ",
    "PU", "BunchCrossing",
    "inTpCharge", "inTpPt",
    "inR", "outR",
]
TARGET_COLS = ["outTpPhi", "outTpEta"]

HDF_PATH = "TTbar_PU50_pixelTracksDoublets_0_final.h5"

NEEDED_COLS = FEATURE_COLS + TARGET_COLS


def load_data(max_samples=None):
    """
    Load the 14 needed columns from the CERN HDF5 file.

    pd.read_hdf fails on Python 3.10+ with the Pandas fixed-format store used
    in this dataset, so we read the two data blocks directly with PyTables.

    When max_samples is set, only that many rows are read from disk (fast path).
    """
    if not os.path.exists(HDF_PATH):
        raise FileNotFoundError(
            f"{HDF_PATH} not found. Download it from CERN Open Data:\n"
            "  http://opendata.cern.ch/eos/opendata/cms/datascience/"
            "CNNPixelSeedsProducerTool/TTbar_13TeV_PU50_PixelSeeds/"
            "TTbar_PU50_pixelTracksDoublets_0_final.h5"
        )

    import tables

    with tables.open_file(HDF_PATH, mode="r") as h5:
        root = h5.root.data
        items0 = [c.decode() for c in root.block0_items[:]]
        items1 = [c.decode() for c in root.block1_items[:]]

        # Determine which columns are needed and their block indices
        need0 = [c for c in NEEDED_COLS if c in items0]
        need1 = [c for c in NEEDED_COLS if c in items1]
        idx0  = [items0.index(c) for c in need0]
        idx1  = [items1.index(c) for c in need1]

        # Read only the rows we need — avoids loading all 5.38M rows when
        # max_samples is small.
        row_slice = slice(None) if max_samples is None else slice(0, max_samples)

        # block*_values shape is (N_rows, N_cols)
        vals0 = root.block0_values[row_slice][:, idx0] if idx0 else np.empty((0, 0))
        vals1 = root.block1_values[row_slice][:, idx1] if idx1 else np.empty((0, 0))

    frames = []
    if idx0:
        frames.append(pd.DataFrame(vals0, columns=need0))
    if idx1:
        frames.append(pd.DataFrame(vals1, columns=need1))
    df = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    missing = [c for c in NEEDED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in HDF5: {missing}")

    df = df[NEEDED_COLS]
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns.")
    return df


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class FCNN(nn.Module):
    """Three-hidden-layer fully connected network with tanh activations."""

    def __init__(self, in_dim, out_dim, hidden=(64, 64, 64)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InversePINN(nn.Module):
    """
    PINN for inverse track-direction prediction from CMS pixel doublets.

    The physics residual is derived from the helical ODE under a solenoidal
    field (B_z ≈ 3.81 T).  Normalization statistics and feature indices are
    stored as buffers so the model is self-contained after training.
    """

    FIXED_BZ = 3.81  # CMS solenoid field [T]

    def __init__(self, in_dim, feature_cols, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.net = FCNN(in_dim, out_dim=2)

        kappa0 = 0.003
        self.kappa_raw = nn.Parameter(
            torch.tensor(math.log(math.exp(kappa0) - 1.0))
        )

        # Normalization stats as buffers (move with .to(device))
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
        self.register_buffer("x_std",  torch.tensor(x_std,  dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(y_std,  dtype=torch.float32))

        # Feature indices stored as plain ints (no grad needed)
        fc = feature_cols
        self.iX  = fc.index("inX");  self.iY  = fc.index("inY");  self.iZ  = fc.index("inZ")
        self.oX  = fc.index("outX"); self.oY  = fc.index("outY"); self.oZ  = fc.index("outZ")
        self.iR  = fc.index("inR");  self.oR  = fc.index("outR")
        self.iq  = fc.index("inTpCharge")
        self.ipt = fc.index("inTpPt")

    @property
    def kappa(self):
        return F.softplus(self.kappa_raw) + 1e-8

    def forward(self, x_norm):
        return self.net(x_norm)

    def pde_residual(self, xb_norm, eps=1e-12):
        """
        Evaluate the helical-motion ODE residual at random collocation
        points τ ∈ [0, 1] sampled fresh each call.

        Returns
        -------
        res        : [N, 2]  (r_cos, r_sin residuals)
        q          : [N]
        valid_pt   : [N] bool mask (pT > 0)
        pt_raw     : [N]
        delta_R    : [N]
        """
        # enable_grad ensures autograd works even when called inside torch.no_grad()
        with torch.enable_grad():
            xb = xb_norm * self.x_std + self.x_mean   # un-normalize to physical

            N = xb.size(0)
            tau = torch.rand(N, device=xb.device, dtype=xb.dtype, requires_grad=True)

            inX  = xb[:, self.iX];  inY  = xb[:, self.iY];  inZ  = xb[:, self.iZ]
            outX = xb[:, self.oX];  outY = xb[:, self.oY];  outZ = xb[:, self.oZ]
            inR  = xb[:, self.iR];  outR0 = xb[:, self.oR]

            dx = outX - inX;  dy = outY - inY;  dz = outZ - inZ
            dR = outR0 - inR

            # Interpolate hit positions and radius along segment
            xi = inX + tau * dx;  yi = inY + tau * dy;  zi = inZ + tau * dz
            Ri = inR + tau * dR

            # Build collocation feature vector without in-place ops so autograd
            # correctly tracks τ through the network forward pass.
            n_feat = xb.shape[1]
            col_map = {self.oX: xi, self.oY: yi, self.oZ: zi, self.oR: Ri}
            cols = [col_map[i].unsqueeze(1) if i in col_map
                    else xb[:, i:i+1].detach() for i in range(n_feat)]
            xc = torch.cat(cols, dim=1)

            # Normalize and forward
            xc_norm = (xc - self.x_mean) / self.x_std
            phi_norm = self.net(xc_norm)[:, 0]
            phi = phi_norm * self.y_std[0] + self.y_mean[0]

            c = torch.cos(phi);  s = torch.sin(phi)

            q      = xb[:, self.iq]
            pt_raw = xb[:, self.ipt]
            valid_pt = pt_raw > 0.0
            pt_safe  = torch.where(valid_pt, pt_raw.clamp(min=1e-2), torch.ones_like(pt_raw))

            sxy       = torch.sqrt(dx**2 + dy**2 + eps)
            omega_tau = sxy * (self.kappa * q * self.FIXED_BZ / (pt_safe + eps))

            dc = torch.autograd.grad(c, tau, grad_outputs=torch.ones_like(c),
                                     create_graph=True, retain_graph=True)[0]
            ds = torch.autograd.grad(s, tau, grad_outputs=torch.ones_like(s),
                                     create_graph=True, retain_graph=True)[0]

            r1 = dc + s * omega_tau
            r2 = ds - c * omega_tau

        return torch.stack([r1, r2], dim=1), q, valid_pt, pt_raw, dR


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def angle_diff(a, b):
    """Smooth, differentiable angular difference ∈ [−π, π]."""
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def pinn_loss_fn(model, xb, yb, lambda_pde_eff,
                 pde_ema, eps=1e-12,
                 update_ema=True, wsum_thresh=10.0,
                 pde_frac=1.0):
    """
    Compute PINN total loss = L_data + λ_pde * L̃_pde.

    pde_frac: fraction of the batch used for PDE residual. Values < 1.0
    trade a noisier physics gradient for proportionally faster epochs.
    Use 1.0 for paper-exact reproducibility.

    Returns
    -------
    total, l_data, l_pde_norm, l_pde_raw, ema_val,
    wsum, skipped, frac_valid, pde_ema  (updated EMA tensor or None)
    """
    preds    = model(xb)
    phi_pred = preds[:, 0] * model.y_std[0] + model.y_mean[0]
    eta_pred = preds[:, 1] * model.y_std[1] + model.y_mean[1]
    phi_true = yb[:, 0] * model.y_std[0] + model.y_mean[0]
    eta_true = yb[:, 1] * model.y_std[1] + model.y_mean[1]

    loss_phi  = torch.mean(angle_diff(phi_pred, phi_true) ** 2)
    loss_eta  = torch.mean((eta_pred - eta_true) ** 2)
    loss_data = loss_eta + loss_phi

    # --- PDE residual ---
    # When lambda_pde_eff == 0 (warmup), skip expensive autograd entirely.
    if lambda_pde_eff == 0.0:
        zero = torch.zeros((), device=xb.device, dtype=xb.dtype)
        if update_ema:
            pass  # leave pde_ema unchanged during warmup
        return (loss_data, loss_data.detach().item(),
                0.0, 0.0, float(pde_ema.item()) if pde_ema is not None else 0.0,
                0.0, True, 0.0, pde_ema)

    # Subsample batch for PDE when pde_frac < 1.0 — reduces autograd cost
    # proportionally while keeping the data loss on the full batch.
    if pde_frac < 1.0:
        n_pde = max(16, int(xb.size(0) * pde_frac))
        idx   = torch.randperm(xb.size(0), device=xb.device)[:n_pde]
        xb_pde = xb[idx]
    else:
        xb_pde = xb

    res, q, valid_pt, pt_raw, dR = model.pde_residual(xb_pde, eps=eps)

    w_pt = valid_pt.float() * torch.sigmoid((pt_raw - 0.5) / 0.5)
    w_dR = torch.tanh(dR.abs() / 2.0)
    w_q  = (q.abs() > 0.8).float()
    w    = w_pt * w_dR * w_q

    pt_safe     = torch.where(valid_pt, pt_raw.clamp(min=1e-2), torch.ones_like(pt_raw))
    omega_scale = torch.sqrt(
        ((dR * model.kappa * q * model.FIXED_BZ / (pt_safe + eps)) ** 2
         * valid_pt.float()).mean()
    ).detach() + 1e-3

    loss_per = (res / omega_scale).pow(2).mean(dim=1)
    wsum     = w.sum()

    if wsum < wsum_thresh:
        loss_pde = torch.zeros((), device=xb.device, dtype=xb.dtype)
        skipped  = True
    else:
        loss_pde = (loss_per * w).sum() / (wsum + 1e-6)
        skipped  = False

    # --- EMA normalization ---
    if update_ema:
        with torch.no_grad():
            v = loss_pde.detach()
            pde_ema = v if pde_ema is None else 0.99 * pde_ema + 0.01 * v

    denom    = (pde_ema if pde_ema is not None else loss_pde.detach() + 1e-8)
    lpde_n   = loss_pde / (denom + 1e-8)
    total    = loss_data + lambda_pde_eff * lpde_n

    return (
        total,
        loss_data.detach().item(),
        lpde_n.detach().item(),             # normalized L_pde (logged in history)
        loss_pde.detach().item(),           # raw L_pde (before EMA normalization)
        float(pde_ema.item()) if pde_ema is not None else 0.0,
        float(wsum.detach()),
        skipped,
        float(valid_pt.float().mean().detach()),
        pde_ema,
    )


def pure_loss_fn(model, xb, yb):
    preds    = model(xb)
    phi_pred = preds[:, 0] * model.y_std[0] + model.y_mean[0]
    eta_pred = preds[:, 1] * model.y_std[1] + model.y_mean[1]
    phi_true = yb[:, 0] * model.y_std[0] + model.y_mean[0]
    eta_true = yb[:, 1] * model.y_std[1] + model.y_mean[1]
    loss = torch.mean(angle_diff(phi_pred, phi_true) ** 2) + \
           torch.mean((eta_pred - eta_true) ** 2)
    return loss


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_pinn(model, X_train, y_train, X_val, y_val, args, device):
    loader_tr = DataLoader(TensorDataset(X_train, y_train),
                           batch_size=args.batch_size, shuffle=True)
    loader_va = DataLoader(TensorDataset(X_val, y_val),
                           batch_size=args.batch_size * 2, shuffle=False)

    opt = optim.Adam(model.net.parameters(), lr=args.lr)

    warmup, ramp = 10, 20
    pde_ema = None

    hist = {k: [] for k in (
        "epoch", "train_loss", "train_data",
        "train_pde",       # EMA-normalized L_pde
        "train_pde_raw",   # raw L_pde (before EMA)
        "pde_ema",         # EMA value
        "val_loss", "val_data", "val_pde", "lambda_pde",
    )}

    for ep in range(1, args.epochs_pinn + 1):
        if ep <= warmup:
            leff = 0.0
        elif ep <= warmup + ramp:
            leff = (ep - warmup) / ramp * args.lambda_pde
        else:
            leff = args.lambda_pde

        # --- train ---
        model.train()
        acc = dict(loss=0., data=0., pde=0., pde_raw=0., ema=0.)
        n_skip = n_bat = 0

        for xb, yb in loader_tr:
            opt.zero_grad()
            (total, ld, lp, lp_raw, ema_val,
             wsum, sk, fv, pde_ema) = pinn_loss_fn(
                model, xb, yb, leff, pde_ema,
                update_ema=True, pde_frac=args.pde_frac
            )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = xb.size(0)
            acc["loss"]    += total.item() * bs
            acc["data"]    += ld * bs
            acc["pde"]     += lp * bs
            acc["pde_raw"] += lp_raw * bs
            acc["ema"]     += ema_val * bs
            n_skip += sk;  n_bat += 1

        N_tr = len(loader_tr.dataset)
        avg  = {k: v / N_tr for k, v in acc.items()}

        # --- val (data loss only — skip expensive PDE autograd) ---
        model.eval()
        vl = vd = vp = 0.
        with torch.no_grad():
            for xb, yb in loader_va:
                preds    = model(xb)
                phi_pred = preds[:, 0] * model.y_std[0] + model.y_mean[0]
                eta_pred = preds[:, 1] * model.y_std[1] + model.y_mean[1]
                phi_true = yb[:, 0] * model.y_std[0] + model.y_mean[0]
                eta_true = yb[:, 1] * model.y_std[1] + model.y_mean[1]
                ld_v = (torch.mean(angle_diff(phi_pred, phi_true) ** 2)
                        + torch.mean((eta_pred - eta_true) ** 2)).item()
                bs = xb.size(0)
                vl += ld_v * bs;  vd += ld_v * bs
        N_va = len(loader_va.dataset)

        hist["epoch"].append(ep)
        hist["train_loss"].append(avg["loss"])
        hist["train_data"].append(avg["data"])
        hist["train_pde"].append(avg["pde"])
        hist["train_pde_raw"].append(avg["pde_raw"])
        hist["pde_ema"].append(avg["ema"])
        hist["val_loss"].append(vl / N_va)
        hist["val_data"].append(vd / N_va)
        hist["val_pde"].append(vp / N_va)
        hist["lambda_pde"].append(leff)

        if ep % 5 == 0 or ep == 1:
            print(
                f"[PINN] Ep {ep:03d} | "
                f"train L={avg['loss']:.3e} "
                f"(data={avg['data']:.3e}, pde={avg['pde']:.3e}, "
                f"pde_raw={avg['pde_raw']:.3e}, ema={avg['ema']:.3e}) | "
                f"val L={vl/N_va:.3e} | "
                f"kappa={model.kappa.item():.5f} | "
                f"λ_eff={leff:.4f} | skip={n_skip}/{n_bat}"
            )

    return hist


def train_pure(model, X_train, y_train, X_val, y_val, args, device):
    loader_tr = DataLoader(TensorDataset(X_train, y_train),
                           batch_size=args.batch_size, shuffle=True)
    loader_va = DataLoader(TensorDataset(X_val, y_val),
                           batch_size=args.batch_size * 2, shuffle=False)

    opt  = optim.Adam(model.parameters(), lr=args.lr)
    hist = {"epoch": [], "train_loss": [], "val_loss": []}

    for ep in range(1, args.epochs_pure + 1):
        model.train()
        total_tr = 0.
        for xb, yb in loader_tr:
            opt.zero_grad()
            loss = pure_loss_fn(model, xb, yb)
            loss.backward()
            opt.step()
            total_tr += loss.item() * xb.size(0)

        model.eval()
        total_va = 0.
        with torch.no_grad():
            for xb, yb in loader_va:
                total_va += pure_loss_fn(model, xb, yb).item() * xb.size(0)

        N_tr = len(loader_tr.dataset);  N_va = len(loader_va.dataset)
        hist["epoch"].append(ep)
        hist["train_loss"].append(total_tr / N_tr)
        hist["val_loss"].append(total_va / N_va)

        if ep % 10 == 0 or ep == 1:
            print(f"[Pure] Ep {ep:03d} | "
                  f"train L={total_tr/N_tr:.3e} | val L={total_va/N_va:.3e}")

    return hist


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, X_tensor):
    model.eval()
    preds_norm = model(X_tensor).cpu().numpy()
    return (preds_norm * model.y_std.cpu().numpy()
            + model.y_mean.cpu().numpy())


def print_metrics(y_true, y_pred, name):
    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    for i, tgt in enumerate(["Phi", "Eta"]):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2  = r2_score(y_true[:, i], y_pred[:, i])
        print(f"  {tgt}: MSE={mse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    df = load_data(args.max_samples)
    X  = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y  = df[TARGET_COLS].to_numpy(dtype=np.float32)

    # --- Normalize ---
    x_mean = X.mean(0);  x_std = X.std(0) + 1e-6
    y_mean = y.mean(0);  y_std = y.std(0) + 1e-6
    X_norm = (X - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # --- Split 70 / 15 / 15 ---
    idx = np.arange(len(X_norm))
    idx_tv, idx_te = train_test_split(idx, test_size=0.15,  random_state=args.seed)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.176, random_state=args.seed)

    def to_t(a): return torch.from_numpy(a).to(device)

    X_tr = to_t(X_norm[idx_tr]);  y_tr = to_t(y_norm[idx_tr])
    X_va = to_t(X_norm[idx_va]);  y_va = to_t(y_norm[idx_va])
    X_te = to_t(X_norm[idx_te]);  y_te = to_t(y_norm[idx_te])
    y_true_te = y[idx_te]

    print(f"Train: {len(idx_tr):,}  Val: {len(idx_va):,}  Test: {len(idx_te):,}")

    # --- Build models ---
    in_dim = X_norm.shape[1]

    pinn_model = InversePINN(
        in_dim, FEATURE_COLS,
        x_mean, x_std, y_mean, y_std
    ).to(device)
    pinn_model.kappa_raw.requires_grad_(False)   # κ frozen during training

    pure_model = InversePINN(
        in_dim, FEATURE_COLS,
        x_mean, x_std, y_mean, y_std
    ).to(device)

    # --- Train ---
    print("\n--- PINN Training ---")
    hist_pinn = train_pinn(pinn_model, X_tr, y_tr, X_va, y_va, args, device)

    print("\n--- Pure NN Training ---")
    hist_pure = train_pure(pure_model, X_tr, y_tr, X_va, y_va, args, device)

    # --- Evaluate ---
    pred_pinn = predict(pinn_model, X_te)
    pred_pure = predict(pure_model, X_te)

    print_metrics(y_true_te, pred_pinn, "PINN — Test Set")
    print_metrics(y_true_te, pred_pure, "Pure NN — Test Set")

    # --- Export df_test.parquet ---
    df_te_raw = df.iloc[idx_te].reset_index(drop=True)

    err_phi_pinn = pred_pinn[:, 0] - y_true_te[:, 0]
    err_phi_pure = pred_pure[:, 0] - y_true_te[:, 0]
    err_eta_pinn = pred_pinn[:, 1] - y_true_te[:, 1]
    err_eta_pure = pred_pure[:, 1] - y_true_te[:, 1]

    df_out = df_te_raw[FEATURE_COLS].copy()
    df_out["phi_true"]     = y_true_te[:, 0]
    df_out["eta_true"]     = y_true_te[:, 1]
    df_out["phi_pinn"]     = pred_pinn[:, 0]
    df_out["eta_pinn"]     = pred_pinn[:, 1]
    df_out["phi_pure"]     = pred_pure[:, 0]
    df_out["eta_pure"]     = pred_pure[:, 1]
    df_out["phi_err_pinn"] = err_phi_pinn
    df_out["eta_err_pinn"] = err_eta_pinn
    df_out["phi_abs_pinn"] = np.abs(err_phi_pinn)
    df_out["eta_abs_pinn"] = np.abs(err_eta_pinn)
    df_out["phi_err_pure"] = err_phi_pure
    df_out["eta_err_pure"] = err_eta_pure
    df_out["phi_abs_pure"] = np.abs(err_phi_pure)
    df_out["eta_abs_pure"] = np.abs(err_eta_pure)

    out_parquet = os.path.join(args.output_dir, "df_test.parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(df_out, preserve_index=False), out_parquet)
    print(f"\nSaved: {out_parquet}  ({len(df_out):,} rows)")

    # --- Export history CSVs ---
    pd.DataFrame(hist_pinn).to_csv(
        os.path.join(args.output_dir, "history_pinn.csv"), index=False)
    pd.DataFrame(hist_pure).to_csv(
        os.path.join(args.output_dir, "history_pure.csv"), index=False)
    print(f"Saved: history_pinn.csv, history_pure.csv")


if __name__ == "__main__":
    main()
