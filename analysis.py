#!/usr/bin/env python3
"""
CMS PINN — Analysis Pipeline

Prerequisites: cms_pinn.py must have been run to produce df_test.parquet.

Run all steps:
    python analysis.py --df_test df_test.parquet

Demo (no Colab needed):
    python analysis.py --df_test df_test_demo.parquet

Select specific steps:
    python analysis.py --steps 1,9,10,11

Steps:
  1  Dataset overview: sizes, sentinel fraction, phi range violations
  2  pT sentinel split: real vs sentinel performance
  3  Kinematic binning: R² by pT, |eta|, dR, charge
  4  Bias scatter: phi error vs phi_true and vs pT  [-> PNG]
  5  Physics residual summary from training logs
  6  Wrap correction effect on phi metrics
  7  L_pde / EMA reconstruction from training history
  8  Barrel analysis: |eta| < 1, eta distribution, dz stats  [-> PNG]
  9  Summary metrics table: MSE, MAE, R² for both models
  10 True vs predicted scatter: phi and eta  [-> PNG]
  11 Error distributions: phi and eta error histograms  [-> PNG]
  12 Training curves: loss vs epoch from history CSVs  [-> PNG]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PI = np.pi


# ── Helpers ──────────────────────────────────────────────────────────────────

def wrap_phi(phi):
    return (phi + PI) % (2 * PI) - PI


def metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2":  r2_score(y_true, y_pred),
    }


def print_table(df, title=""):
    sep = "═" * max(60, len(title) + 4)
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(df.to_string(index=False))
    print(sep)


def ensure_err_cols(df):
    for m, pc, ec in [("pinn", "phi_pinn", "eta_pinn"),
                      ("pure", "phi_pure", "eta_pure")]:
        if f"phi_err_{m}" not in df.columns:
            df[f"phi_err_{m}"] = df[pc] - df["phi_true"]
        if f"eta_err_{m}" not in df.columns:
            df[f"eta_err_{m}"] = df[ec] - df["eta_true"]
    return df


# ── Step 1 — Dataset Overview ─────────────────────────────────────────────────

def step1(df, n_total=None, n_train=None, n_val=None):
    N = len(df)
    n_neg_pt   = (df["inTpPt"] == -1.0).sum()
    n_pinn_out = ((df["phi_pinn"] < -PI) | (df["phi_pinn"] > PI)).sum()
    n_pure_out = ((df["phi_pure"] < -PI) | (df["phi_pure"] > PI)).sum()

    if n_total and n_train and n_val:
        size_rows = [
            ["Total",  n_total,  f"{n_total/n_total*100:.2f}%"],
            ["Train",  n_train,  f"{n_train/n_total*100:.2f}%"],
            ["Val",    n_val,    f"{n_val/n_total*100:.2f}%"],
            ["Test",   N,        f"{N/n_total*100:.2f}%"],
        ]
        print_table(pd.DataFrame(size_rows, columns=["Split", "N", "Fraction"]),
                    "STEP 1a — Dataset Sizes")

    rows = [
        ["Test size",              N,          "—"],
        ["inTpPt = -1.0 (fake)",   n_neg_pt,   f"{n_neg_pt/N*100:.2f}%"],
        ["inTpPt > 0  (real)",     N-n_neg_pt, f"{(N-n_neg_pt)/N*100:.2f}%"],
        ["PINN phi out of [-pi,pi]",  n_pinn_out, f"{n_pinn_out/N*100:.2f}%"],
        ["Pure NN phi out of [-pi,pi]",n_pure_out, f"{n_pure_out/N*100:.2f}%"],
    ]
    print_table(pd.DataFrame(rows, columns=["Metric", "N", "Fraction"]),
                "STEP 1b — Test Set Summary")


# ── Step 2 — pT Sentinel Split ────────────────────────────────────────────────

def step2(df):
    mask_fake = df["inTpPt"] == -1.0
    rows = []
    for label, mask in [("inTpPt = -1.0 (fake)", mask_fake),
                        ("inTpPt > 0   (real)",  ~mask_fake)]:
        sub = df[mask]
        if sub.empty:
            continue
        for model, pc, ec in [("PINN",    "phi_pinn", "eta_pinn"),
                               ("Pure NN", "phi_pure", "eta_pure")]:
            mp = metrics(sub["phi_true"], sub[pc])
            me = metrics(sub["eta_true"], sub[ec])
            rows.append([label, model, f"{len(sub):,}",
                         f"{mp['MSE']:.4f}", f"{mp['MAE']:.4f}", f"{mp['R2']:.4f}",
                         f"{me['MSE']:.4f}", f"{me['MAE']:.4f}", f"{me['R2']:.4f}"])
    tbl = pd.DataFrame(rows, columns=["Group","Model","N",
                                       "phi_MSE","phi_MAE","phi_R2",
                                       "eta_MSE","eta_MAE","eta_R2"])
    print_table(tbl, "STEP 2 — pT Sentinel Split")
    return tbl


# ── Step 3 — Kinematic Binning ────────────────────────────────────────────────

def step3(df):
    df = df.copy()
    df["abs_eta_true"] = df["eta_true"].abs()
    df["dR"] = df["outR"] - df["inR"]

    def _bin_table(df, bin_col, bins, labels, title):
        rows = []
        for (lo, hi), lbl in zip(bins, labels):
            sub = df[df[bin_col] >= lo] if hi is None else df[(df[bin_col] >= lo) & (df[bin_col] < hi)]
            if len(sub) < 10:
                continue
            for model, pc, ec in [("PINN",    "phi_pinn", "eta_pinn"),
                                   ("Pure NN", "phi_pure", "eta_pure")]:
                rows.append([lbl, model, f"{len(sub):,}",
                              round(r2_score(sub["phi_true"], sub[pc]), 4),
                              round(r2_score(sub["eta_true"], sub[ec]), 4)])
        tbl = pd.DataFrame(rows, columns=["Bin","Model","N","R2_phi","R2_eta"])
        print_table(tbl, title)
        return tbl

    df_pos = df[df["inTpPt"] > 0]
    _bin_table(df_pos, "inTpPt",
               [(0,1),(1,2),(2,5),(5,None)], ["[0,1)","[1,2)","[2,5)","[5,inf)"],
               "STEP 3a — R² by pT bin (inTpPt > 0 only)")

    _bin_table(df, "abs_eta_true",
               [(0,1),(1,2),(2,2.5)], ["[0,1)","[1,2)","[2,2.5)"],
               "STEP 3b — R² by |eta| bin")

    rows_q = []
    for q in [1.0, -1.0]:
        sub = df[df["inTpCharge"] == q]
        if len(sub) < 10:
            continue
        for model, pc, ec in [("PINN","phi_pinn","eta_pinn"),("Pure NN","phi_pure","eta_pure")]:
            rows_q.append([f"q={q:+.0f}", model, f"{len(sub):,}",
                           round(r2_score(sub["phi_true"], sub[pc]), 4),
                           round(r2_score(sub["eta_true"], sub[ec]), 4)])
    print_table(pd.DataFrame(rows_q, columns=["Charge","Model","N","R2_phi","R2_eta"]),
                "STEP 3c — R² by Charge")

    q33, q66 = df["dR"].quantile([1/3, 2/3]).values
    _bin_table(df, "dR",
               [(df["dR"].min(), q33),(q33,q66),(q66,None)],
               [f"small  (<{q33:.1f}cm)", f"mid    [{q33:.1f},{q66:.1f})cm", f"large  (>={q66:.1f}cm)"],
               "STEP 3d — R² by dR bin (dR = outR - inR)")


# ── Step 4 — Bias Scatter Plots ───────────────────────────────────────────────

def step4(df, out_dir="."):
    df = ensure_err_cols(df)
    n   = min(200_000, len(df))
    sub = df.sample(n=n, random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (model, col) in zip(axes, [("PINN","phi_err_pinn"),("Pure NN","phi_err_pure")]):
        x = sub["phi_true"].values;  y = sub[col].values
        ax.scatter(x, y, s=1, alpha=0.15, rasterized=True, color="steelblue")
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 300)
        ax.plot(xr, m*xr+b, "r-", lw=2, label=f"y={m:.3f}x+{b:.3f}")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("True phi (rad)");  ax.set_ylabel("phi error (pred - true)")
        ax.set_title(f"{model}: phi error vs True phi");  ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.suptitle("STEP 4a — Bias: phi error vs True phi", fontsize=13)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "step4a_phi_err_vs_phi_true.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight");  plt.close()

    sub_pos = sub[sub["inTpPt"] > 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (model, col) in zip(axes, [("PINN","phi_err_pinn"),("Pure NN","phi_err_pure")]):
        x = sub_pos["inTpPt"].values;  y = sub_pos[col].values
        ax.scatter(x, y, s=1, alpha=0.15, rasterized=True, color="darkorange")
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(0, min(x.max(), 25), 300)
        ax.plot(xr, m*xr+b, "r-", lw=2, label=f"y={m:.4f}x+{b:.3f}")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("inTpPt (GeV/c)");  ax.set_ylabel("phi error (pred - true)")
        ax.set_title(f"{model}: phi error vs pT");  ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.suptitle("STEP 4b — Bias: phi error vs pT", fontsize=13)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "step4b_phi_err_vs_pt.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight");  plt.close()

    print(f"\n[STEP 4] Saved: {p1}\n        Saved: {p2}")


# ── Step 5 — Physics Residual (text) ─────────────────────────────────────────

def step5(hist_pinn_path=None):
    print("\n" + "═"*60)
    print("  STEP 5 — Physics Residual (Training Logs)")
    print("═"*60)

    if hist_pinn_path and os.path.exists(hist_pinn_path):
        hist = pd.read_csv(hist_pinn_path)
        cols_want = ["epoch","train_data","train_pde","train_loss","val_data","val_pde","val_loss"]
        cols_have = [c for c in cols_want if c in hist.columns]
        last10 = hist.tail(10)[cols_have].copy()
        print(last10.round(5).to_string(index=False))
        avg_data = hist.tail(10)["train_data"].mean() if "train_data" in hist.columns else float("nan")
        avg_pde  = hist.tail(10)["train_pde"].mean()  if "train_pde"  in hist.columns else float("nan")
        avg_tot  = hist.tail(10)["train_loss"].mean() if "train_loss" in hist.columns else float("nan")
        print(f"\nAvg last 10 epochs:  L_data={avg_data:.5f}  L_pde={avg_pde:.5f}  L_total={avg_tot:.5f}")
    else:
        print("history_pinn.csv not found — run cms_pinn.py first, or use --hist_pinn flag.")
    print("═"*60)


# ── Step 6 — Wrap Correction ─────────────────────────────────────────────────

def step6(df):
    rows = []
    for model, col in [("PINN","phi_pinn"),("Pure NN","phi_pure")]:
        y_true = df["phi_true"].values
        y_raw  = df[col].values
        y_wrap = wrap_phi(y_raw)
        n_out_raw  = int(((y_raw  < -PI) | (y_raw  > PI)).sum())
        n_out_wrap = int(((y_wrap < -PI) | (y_wrap > PI)).sum())
        m_raw  = metrics(y_true, y_raw)
        m_wrap = metrics(y_true, y_wrap)
        rows += [
            [model, "Raw",           n_out_raw,  m_raw["MSE"],  m_raw["MAE"],  m_raw["R2"]],
            [model, "After wrap",    n_out_wrap, m_wrap["MSE"], m_wrap["MAE"], m_wrap["R2"]],
        ]
    tbl = pd.DataFrame(rows, columns=["Model","State","phi_out_of_range","MSE","MAE","R2"])
    tbl[["MSE","MAE","R2"]] = tbl[["MSE","MAE","R2"]].round(6)
    print_table(tbl, "STEP 6 — Wrap Correction Effect on phi Metrics")
    return tbl


# ── Step 7 — L_pde / EMA (text) ──────────────────────────────────────────────

def step7(hist_pinn_path=None):
    print("\n" + "═"*65)
    print("  STEP 7 — Normalized vs Raw L_pde")
    print("═"*65)

    if hist_pinn_path and os.path.exists(hist_pinn_path):
        hist = pd.read_csv(hist_pinn_path)
        if "train_pde_raw" in hist.columns and "pde_ema" in hist.columns:
            last10 = hist.tail(10)[["epoch","train_pde","train_pde_raw","pde_ema"]].copy()
            last10.columns = ["Epoch","L_pde_norm","L_pde_raw","EMA"]
            print(last10.round(6).to_string(index=False))
            avg = hist.tail(10)[["train_pde","train_pde_raw","pde_ema"]].mean()
            print(f"\nAvg last 10 epochs:")
            print(f"  L_pde_norm = {avg['train_pde']:.5f}  (EMA-normalized)")
            print(f"  L_pde_raw  = {avg['train_pde_raw']:.5f}  (before EMA)")
            print(f"  EMA        = {avg['pde_ema']:.5f}")
            print(f"  raw/EMA    = {avg['train_pde_raw']/avg['pde_ema']:.4f}  (-> 1.0 at convergence)")
        else:
            print("Columns 'train_pde_raw' / 'pde_ema' not found in history_pinn.csv.")
            print("These are logged by cms_pinn.py — check that you're using the current version.")
    else:
        print("history_pinn.csv not found — run cms_pinn.py first, or use --hist_pinn flag.")
    print("═"*65)


# ── Step 8 — Barrel Analysis ─────────────────────────────────────────────────

def step8(df, out_dir="."):
    mask = df["eta_true"].abs() < 1.0
    sub  = df[mask].copy()
    n    = len(sub)

    print("\n" + "═"*65)
    print(f"  STEP 8 — Barrel Region |eta| < 1  (N = {n:,})")
    print("═"*65)

    sub = ensure_err_cols(sub)

    rows = []
    for label, vals in [
        ("eta_true",     sub["eta_true"].values),
        ("eta_pinn",     sub["eta_pinn"].values),
        ("eta_pure",     sub["eta_pure"].values),
        ("eta_err_pinn", sub["eta_err_pinn"].values),
        ("eta_err_pure", sub["eta_err_pure"].values),
    ]:
        rows.append({"Variable": label,
                     "mean":   round(float(np.mean(vals)),  4),
                     "std":    round(float(np.std(vals)),   4),
                     "p05":    round(float(np.percentile(vals,  5)), 4),
                     "median": round(float(np.median(vals)),        4),
                     "p95":    round(float(np.percentile(vals, 95)), 4)})
    print_table(pd.DataFrame(rows), "8a — eta Distribution (|eta_true| < 1)")

    metric_rows = []
    for model, pc, ec in [("PINN","phi_pinn","eta_pinn"),("Pure NN","phi_pure","eta_pure")]:
        mp = metrics(sub["phi_true"], sub[pc])
        me = metrics(sub["eta_true"], sub[ec])
        metric_rows.append([model, f"{n:,}",
                             f"{mp['MSE']:.4f}", f"{mp['R2']:.4f}",
                             f"{me['MSE']:.4f}", f"{me['R2']:.4f}"])
    print_table(pd.DataFrame(metric_rows, columns=["Model","N","phi_MSE","phi_R2","eta_MSE","eta_R2"]),
                "8b — Metrics (|eta_true| < 1)")

    dz = sub["outZ"].values - sub["inZ"].values
    dz_out = df.loc[df["eta_true"].abs() >= 1.0, "outZ"].values - \
             df.loc[df["eta_true"].abs() >= 1.0, "inZ"].values

    print(f"\n  8c — dz = outZ - inZ")
    print(f"  Barrel  |eta|<1  : mean={np.mean(dz):+.3f}  std={np.std(dz):.3f} cm  (N={n:,})")
    print(f"  Endcap  |eta|>=1 : mean={np.mean(dz_out):+.3f}  std={np.std(dz_out):.3f} cm  (N={len(dz_out):,})")
    print(f"\n  eta_R2 barrel: PINN={r2_score(sub['eta_true'],sub['eta_pinn']):.4f}  "
          f"Pure NN={r2_score(sub['eta_true'],sub['eta_pure']):.4f}")
    print(f"  Small dz in barrel ({np.std(dz):.1f} cm) gives little z-leverage for eta estimation.")
    print("═"*65)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, vals, color) in zip(axes, [
        ("eta_true",    sub["eta_true"].values, "black"),
        ("eta PINN",    sub["eta_pinn"].values, "orange"),
        ("eta Pure NN", sub["eta_pure"].values, "steelblue"),
    ]):
        ax.hist(vals, bins=60, color=color, alpha=0.7, density=True)
        ax.axvline(np.mean(vals), color="red", ls="--", lw=1.5,
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_title(f"{label}  (|eta_true|<1)", fontsize=11)
        ax.set_xlabel("eta");  ax.set_ylabel("Density")
        ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
    plt.suptitle("STEP 8 — eta Distribution: True vs Predicted  (|eta_true| < 1)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "step8_eta_dist_barrel.png")
    plt.savefig(p, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n[STEP 8] Saved: {p}")


# ── Step 9 — Summary Metrics Table ───────────────────────────────────────────

def step9(df):
    """Clean MSE/MAE/R² table for both models, both targets, full test set."""
    rows = []
    for model, pc, ec in [("PINN", "phi_pinn", "eta_pinn"),
                           ("Pure NN", "phi_pure", "eta_pure")]:
        mp = metrics(df["phi_true"], df[pc])
        me = metrics(df["eta_true"], df[ec])
        rows.append([model,
                     f"{mp['MSE']:.4f}", f"{mp['MAE']:.4f}", f"{mp['R2']:.4f}",
                     f"{me['MSE']:.4f}", f"{me['MAE']:.4f}", f"{me['R2']:.4f}"])
    tbl = pd.DataFrame(rows, columns=["Model",
                                       "phi_MSE","phi_MAE","phi_R2",
                                       "eta_MSE","eta_MAE","eta_R2"])
    print_table(tbl, "STEP 9 — Summary Metrics (Full Test Set)")

    # Delta
    row_pinn = rows[0];  row_pure = rows[1]
    print("  Improvement of Pure NN over PINN:")
    for i, col in enumerate(["phi_MSE","phi_MAE","phi_R2","eta_MSE","eta_MAE","eta_R2"], start=1):
        v_pinn = float(rows[0][i]);  v_pure = float(rows[1][i])
        if col.endswith("R2"):
            print(f"    {col}: PINN={v_pinn:.4f}  PureNN={v_pure:.4f}  delta={v_pure-v_pinn:+.4f}")
        else:
            pct = (v_pinn - v_pure) / v_pinn * 100
            print(f"    {col}: PINN={v_pinn:.4f}  PureNN={v_pure:.4f}  reduction={pct:+.1f}%")
    return tbl


# ── Step 10 — True vs Predicted Scatter ──────────────────────────────────────

def step10(df, out_dir="."):
    """2×2 scatter: true vs predicted for phi and eta, PINN and Pure NN."""
    n   = min(80_000, len(df))
    sub = df.sample(n=n, random_state=42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    target_configs = [
        ("phi", "phi_true", "phi_pinn", "phi_pure", "phi (rad)", [-PI, PI], [-PI, PI]),
        ("eta", "eta_true", "eta_pinn", "eta_pure", "eta",       [-3.5, 3.5], [-6, 6]),
    ]

    for row_idx, (target, tc, pc_pinn, pc_pure, xlabel, xlim, ylim) in enumerate(target_configs):
        for col_idx, (model, pred_col) in enumerate([("PINN", pc_pinn), ("Pure NN", pc_pure)]):
            ax = axes[row_idx][col_idx]
            x  = sub[tc].values
            y  = sub[pred_col].values

            ax.scatter(x, y, s=1, alpha=0.1, rasterized=True,
                       color="steelblue" if col_idx == 0 else "darkorange")

            # Diagonal (perfect prediction)
            lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
            ax.plot(lim, lim, "r--", lw=1.5, alpha=0.8, label="perfect")

            r2 = r2_score(sub[tc], sub[pred_col])
            ax.set_xlabel(f"True {xlabel}", fontsize=10)
            ax.set_ylabel(f"Predicted {xlabel}", fontsize=10)
            ax.set_title(f"{model} — {target}  (R²={r2:.3f})", fontsize=11)
            ax.set_xlim(xlim);  ax.set_ylim(ylim)
            ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    plt.suptitle("STEP 10 — True vs Predicted  (sample)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "step10_true_vs_pred.png")
    plt.savefig(p, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n[STEP 10] Saved: {p}")


# ── Step 11 — Error Distributions ────────────────────────────────────────────

def step11(df, out_dir="."):
    """Histograms of phi and eta prediction errors for both models."""
    df = ensure_err_cols(df)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    configs = [
        (0, 0, "phi_err_pinn", "PINN — phi error",    "steelblue"),
        (0, 1, "phi_err_pure", "Pure NN — phi error", "darkorange"),
        (1, 0, "eta_err_pinn", "PINN — eta error",    "steelblue"),
        (1, 1, "eta_err_pure", "Pure NN — eta error", "darkorange"),
    ]

    for row, col, err_col, title, color in configs:
        ax  = axes[row][col]
        err = df[err_col].values

        # Clip extreme outliers for readability
        p1, p99 = np.percentile(err, [1, 99])
        ax.hist(err, bins=100, range=(p1, p99), color=color,
                alpha=0.75, density=True)

        mean_e = np.mean(err);  std_e = np.std(err)
        ax.axvline(0,      color="black", ls="--", lw=1.5, label="zero")
        ax.axvline(mean_e, color="red",   ls="-",  lw=1.5,
                   label=f"mean={mean_e:+.3f}")

        # Gaussian fit overlay
        from scipy.stats import norm as _norm
        xs = np.linspace(p1, p99, 300)
        ax.plot(xs, _norm.pdf(xs, mean_e, std_e), "k-", lw=1.5, alpha=0.7,
                label=f"N(μ,σ)  σ={std_e:.3f}")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Error (pred - true)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    plt.suptitle("STEP 11 — Error Distributions (1–99th percentile shown)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "step11_error_distributions.png")
    plt.savefig(p, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n[STEP 11] Saved: {p}")

    # Print numeric summary
    print("\n  Error statistics (full test set):")
    header = f"  {'Metric':<22} {'mean':>8} {'std':>8} {'p25':>8} {'median':>8} {'p75':>8}"
    print(header)
    print("  " + "-"*62)
    for col in ["phi_err_pinn","phi_err_pure","eta_err_pinn","eta_err_pure"]:
        e = df[col].values
        print(f"  {col:<22} {np.mean(e):>+8.4f} {np.std(e):>8.4f} "
              f"{np.percentile(e,25):>+8.4f} {np.median(e):>+8.4f} "
              f"{np.percentile(e,75):>+8.4f}")


# ── Step 12 — Training Curves ─────────────────────────────────────────────────

def step12(hist_pinn_path=None, hist_pure_path=None, out_dir="."):
    """Loss vs epoch plots for PINN and Pure NN."""
    pinn_ok = hist_pinn_path and os.path.exists(hist_pinn_path)
    pure_ok = hist_pure_path and os.path.exists(hist_pure_path)

    if not pinn_ok and not pure_ok:
        print("\n[STEP 12] history_pinn.csv and history_pure.csv not found — skipping.")
        print("  Run cms_pinn.py first to generate training history files.")
        return

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

    # ── 1) Total loss: PINN vs Pure NN ──
    ax1 = fig.add_subplot(gs[0, :2])
    if pinn_ok:
        hp = pd.read_csv(hist_pinn_path)
        ax1.plot(hp["epoch"], hp["train_loss"], color="steelblue",  lw=1.5, label="PINN train")
        ax1.plot(hp["epoch"], hp["val_loss"],   color="steelblue",  lw=1.5, ls="--", alpha=0.7, label="PINN val")
    if pure_ok:
        hq = pd.read_csv(hist_pure_path)
        ax1.plot(hq["epoch"], hq["train_loss"], color="darkorange", lw=1.5, label="Pure NN train")
        ax1.plot(hq["epoch"], hq["val_loss"],   color="darkorange", lw=1.5, ls="--", alpha=0.7, label="Pure NN val")
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Total loss")
    ax1.set_title("Total Loss: PINN vs Pure NN", fontsize=11)
    ax1.legend(fontsize=9);  ax1.grid(True, alpha=0.3)

    # ── 2) PINN loss breakdown ──
    if pinn_ok and "train_data" in hp.columns:
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(hp["epoch"], hp["train_data"], color="green",      lw=1.5, label="L_data")
        ax2.plot(hp["epoch"], hp["train_pde"],  color="purple",     lw=1.5, label="L_pde (norm)")
        if "train_pde_raw" in hp.columns:
            ax2.plot(hp["epoch"], hp["train_pde_raw"], color="red", lw=1.0, ls=":", label="L_pde (raw)")
        ax2.set_xlabel("Epoch");  ax2.set_ylabel("Loss component")
        ax2.set_title("PINN Loss Components", fontsize=11)
        ax2.legend(fontsize=8);  ax2.grid(True, alpha=0.3)

    # ── 3) λ_pde schedule ──
    if pinn_ok and "lambda_pde" in hp.columns:
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(hp["epoch"], hp["lambda_pde"], color="brown", lw=2)
        ax3.set_xlabel("Epoch");  ax3.set_ylabel("lambda_pde (effective)")
        ax3.set_title("lambda_pde Schedule (warmup + ramp)", fontsize=11)
        ax3.grid(True, alpha=0.3)

    # ── 4) EMA evolution ──
    if pinn_ok and "pde_ema" in hp.columns:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(hp["epoch"], hp["pde_ema"], color="teal", lw=2, label="EMA")
        if "train_pde_raw" in hp.columns:
            ax4.plot(hp["epoch"], hp["train_pde_raw"], color="red", lw=1, alpha=0.6, label="L_pde_raw")
        ax4.set_xlabel("Epoch");  ax4.set_ylabel("Value")
        ax4.set_title("L_pde_raw vs EMA over Training", fontsize=11)
        ax4.legend(fontsize=8);  ax4.grid(True, alpha=0.3)

    # ── 5) Val loss comparison (log scale) ──
    ax5 = fig.add_subplot(gs[1, 2])
    if pinn_ok:
        ax5.semilogy(hp["epoch"], hp["val_loss"], color="steelblue",  lw=1.5, label="PINN val")
    if pure_ok:
        ax5.semilogy(hq["epoch"], hq["val_loss"], color="darkorange", lw=1.5, label="Pure NN val")
    ax5.set_xlabel("Epoch");  ax5.set_ylabel("Val loss (log scale)")
    ax5.set_title("Validation Loss (log)", fontsize=11)
    ax5.legend(fontsize=9);  ax5.grid(True, alpha=0.3)

    p = os.path.join(out_dir, "step12_training_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight");  plt.close()
    print(f"\n[STEP 12] Saved: {p}")

    # Print final epoch summary
    if pinn_ok:
        final = hp.iloc[-1]
        print(f"\n  PINN final epoch {int(final['epoch'])}:")
        print(f"    train_loss={final['train_loss']:.5f}  val_loss={final['val_loss']:.5f}")
        if "train_data" in hp.columns:
            print(f"    train_data={final['train_data']:.5f}  train_pde={final['train_pde']:.5f}")
        if "train_pde_raw" in hp.columns:
            print(f"    train_pde_raw={final['train_pde_raw']:.5f}  pde_ema={final['pde_ema']:.5f}")
    if pure_ok:
        final = hq.iloc[-1]
        print(f"\n  Pure NN final epoch {int(final['epoch'])}:")
        print(f"    train_loss={final['train_loss']:.5f}  val_loss={final['val_loss']:.5f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CMS PINN Analysis Pipeline")
    parser.add_argument("--df_test",   default="df_test.parquet")
    parser.add_argument("--hist_pinn", default="history_pinn.csv")
    parser.add_argument("--hist_pure", default="history_pure.csv")
    parser.add_argument("--out_dir",   default="figures")
    parser.add_argument("--steps",     default="1,2,3,4,5,6,7,8,9,10,11,12",
                        help="Comma-separated list of steps to run")
    parser.add_argument("--n_total",   type=int, default=None)
    parser.add_argument("--n_train",   type=int, default=None)
    parser.add_argument("--n_val",     type=int, default=None)
    args = parser.parse_args()

    steps = {int(s.strip()) for s in args.steps.split(",")}

    if not os.path.exists(args.df_test):
        print(f"\nERROR: '{args.df_test}' not found.")
        print("Run cms_pinn.py first to generate df_test.parquet, then download from Colab:")
        print("  df_test.to_parquet('df_test.parquet', index=False)")
        print("  from google.colab import files; files.download('df_test.parquet')")
        return

    print(f"\nLoading: {args.df_test}")
    df = pd.read_parquet(args.df_test)
    print(f"  {len(df):,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    if 1  in steps: step1(df, n_total=args.n_total, n_train=args.n_train, n_val=args.n_val)
    if 2  in steps: step2(df)
    if 3  in steps: step3(df)
    if 4  in steps: step4(df, out_dir=args.out_dir)
    if 5  in steps: step5(hist_pinn_path=args.hist_pinn)
    if 6  in steps: step6(df)
    if 7  in steps: step7(hist_pinn_path=args.hist_pinn)
    if 8  in steps: step8(df, out_dir=args.out_dir)
    if 9  in steps: step9(df)
    if 10 in steps: step10(df, out_dir=args.out_dir)
    if 11 in steps: step11(df, out_dir=args.out_dir)
    if 12 in steps: step12(hist_pinn_path=args.hist_pinn,
                           hist_pure_path=args.hist_pure,
                           out_dir=args.out_dir)

    print(f"\n✓ Done. Figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
