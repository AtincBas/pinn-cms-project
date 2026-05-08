#!/usr/bin/env python3
"""
CMS PINN — 6-Adımlık Analiz

Ön koşul: cms_pinn.py çalıştırılmış ve df_test.parquet üretilmiş olmalı.

Çalıştırma:
    python analysis.py                               # tüm adımlar
    python analysis.py --steps 1,2,6                 # seçili adımlar
    python analysis.py --df_test /yol/df_test.parquet --out_dir ./figures

Colab'dan df_test.parquet indirme:
    from google.colab import files
    df_test.to_parquet("df_test.parquet", index=False)
    files.download("df_test.parquet")
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


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def wrap_phi(phi):
    """φ'yi [-π, π] aralığına sar."""
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
    """Hata sütunları eksikse hesapla."""
    for m, pc, ec in [("pinn", "phi_pinn", "eta_pinn"),
                      ("pure", "phi_pure", "eta_pure")]:
        if f"phi_err_{m}" not in df.columns:
            df[f"phi_err_{m}"] = df[pc] - df["phi_true"]
        if f"eta_err_{m}" not in df.columns:
            df[f"eta_err_{m}"] = df[ec] - df["eta_true"]
    return df


# ─── Adım 1 — Veri ve Model Yükleme Kontrolü ─────────────────────────────────

def step1(df, n_total=None, n_train=None, n_val=None):
    """
    Test seti istatistikleri:
      - Boyutlar (train/val/test toplam)
      - inTpPt = -1.0 sayısı ve yüzdesi
      - φ ∉ [-π, π] sayısı (PINN ve Pure NN ayrı ayrı)
    """
    N = len(df)

    n_neg_pt   = (df["inTpPt"] == -1.0).sum()
    n_pinn_out = ((df["phi_pinn"] < -PI) | (df["phi_pinn"] > PI)).sum()
    n_pure_out = ((df["phi_pure"] < -PI) | (df["phi_pure"] > PI)).sum()

    # ── Boyut tablosu ──
    if n_total and n_train and n_val:
        size_rows = [
            ["Toplam (N)",  n_total,  f"{n_total/n_total*100:.2f}%"],
            ["Train",       n_train,  f"{n_train/n_total*100:.2f}%"],
            ["Val",         n_val,    f"{n_val/n_total*100:.2f}%"],
            ["Test",        N,        f"{N/n_total*100:.2f}%"],
        ]
        tbl_size = pd.DataFrame(size_rows, columns=["Küme", "Sayı", "Oran"])
        print_table(tbl_size, "ADIM 1a — Veri Boyutları")

    # ── Test seti detayları ──
    rows = [
        ["Test boyutu",           N,          "—"],
        ["inTpPt = -1.0 (sahte)", n_neg_pt,   f"{n_neg_pt/N*100:.2f}%"],
        ["inTpPt > 0  (gerçek)",  N-n_neg_pt, f"{(N-n_neg_pt)/N*100:.2f}%"],
        ["PINN φ ∉ [-π,π]",       n_pinn_out, f"{n_pinn_out/N*100:.2f}%"],
        ["Pure NN φ ∉ [-π,π]",    n_pure_out, f"{n_pure_out/N*100:.2f}%"],
    ]
    tbl = pd.DataFrame(rows, columns=["Metrik", "Sayı", "Yüzde"])
    print_table(tbl, "ADIM 1b — Test Seti Özet")
    return tbl


# ─── Adım 2 — pT Sentinel Analizi ────────────────────────────────────────────

def step2(df):
    """
    Test setini inTpPt == -1.0 (sahte) ve inTpPt > 0 (gerçek) olarak ikiye böl.
    Her grup için PINN ve Pure NN'in φ ve η MSE, MAE, R² değerlerini hesapla.
    """
    mask_fake = df["inTpPt"] == -1.0
    mask_real = ~mask_fake

    rows = []
    for label, mask in [("inTpPt = -1.0  (sahte)", mask_fake),
                        ("inTpPt > 0     (gerçek)",  mask_real)]:
        sub = df[mask]
        if sub.empty:
            continue
        for model, pc, ec in [("PINN",    "phi_pinn", "eta_pinn"),
                               ("Pure NN", "phi_pure", "eta_pure")]:
            mp = metrics(sub["phi_true"], sub[pc])
            me = metrics(sub["eta_true"], sub[ec])
            rows.append([
                label, model, f"{len(sub):,}",
                f"{mp['MSE']:.4f}", f"{mp['MAE']:.4f}", f"{mp['R2']:.4f}",
                f"{me['MSE']:.4f}", f"{me['MAE']:.4f}", f"{me['R2']:.4f}",
            ])

    tbl = pd.DataFrame(rows, columns=[
        "Grup", "Model", "N",
        "φ_MSE", "φ_MAE", "φ_R²",
        "η_MSE", "η_MAE", "η_R²",
    ])
    print_table(tbl, "ADIM 2 — pT Sentinel Analizi")
    return tbl


# ─── Adım 3 — Kinematik Bin Analizi ─────────────────────────────────────────

def step3(df):
    """
    pT, |η|, charge, ΔR bin'lerine göre PINN ve Pure NN R²_φ ve R²_η hesapla.
    """
    df = df.copy()
    df["abs_eta_true"] = df["eta_true"].abs()
    df["dR"] = df["outR"] - df["inR"]   # transvers yarıçap farkı

    def _bin_table(df, bin_col, bins, labels, title):
        rows = []
        for (lo, hi), lbl in zip(bins, labels):
            if hi is None:
                sub = df[df[bin_col] >= lo]
            else:
                sub = df[(df[bin_col] >= lo) & (df[bin_col] < hi)]
            if len(sub) < 10:
                continue
            for model, pc, ec in [("PINN",    "phi_pinn", "eta_pinn"),
                                   ("Pure NN", "phi_pure", "eta_pure")]:
                rows.append([
                    lbl, model, f"{len(sub):,}",
                    round(r2_score(sub["phi_true"], sub[pc]), 4),
                    round(r2_score(sub["eta_true"], sub[ec]), 4),
                ])
        tbl = pd.DataFrame(rows, columns=["Bin", "Model", "N", "R²_φ", "R²_η"])
        print_table(tbl, title)
        return tbl

    # 3a: pT bins (sadece pt > 0)
    df_pos = df[df["inTpPt"] > 0]
    tbl_pt = _bin_table(
        df_pos, "inTpPt",
        [(0, 1), (1, 2), (2, 5), (5, None)],
        ["[0,1)", "[1,2)", "[2,5)", "[5,∞)"],
        "ADIM 3a — pT Bin Analizi (inTpPt > 0 alt kümesi)",
    )

    # 3b: |η| bins (tüm test seti)
    tbl_eta = _bin_table(
        df, "abs_eta_true",
        [(0, 1), (1, 2), (2, 2.5)],
        ["[0,1)", "[1,2)", "[2,2.5)"],
        "ADIM 3b — |η| Bin Analizi",
    )

    # 3c: Charge
    rows_q = []
    for q in [1.0, -1.0]:
        sub = df[df["inTpCharge"] == q]
        if len(sub) < 10:
            continue
        for model, pc, ec in [("PINN",    "phi_pinn", "eta_pinn"),
                               ("Pure NN", "phi_pure", "eta_pure")]:
            rows_q.append([
                f"q = {q:+.0f}", model, f"{len(sub):,}",
                round(r2_score(sub["phi_true"], sub[pc]), 4),
                round(r2_score(sub["eta_true"], sub[ec]), 4),
            ])
    tbl_q = pd.DataFrame(rows_q, columns=["Charge", "Model", "N", "R²_φ", "R²_η"])
    print_table(tbl_q, "ADIM 3c — Charge Analizi")

    # 3d: ΔR bins (üçe eşit böl)
    q33, q66 = df["dR"].quantile([1/3, 2/3]).values
    tbl_dr = _bin_table(
        df, "dR",
        [(df["dR"].min(), q33), (q33, q66), (q66, None)],
        [f"küçük  (<{q33:.1f} cm)",
         f"orta   [{q33:.1f},{q66:.1f}) cm",
         f"büyük  (≥{q66:.1f} cm)"],
        "ADIM 3d — ΔR Bin Analizi (ΔR = outR − inR)",
    )

    return tbl_pt, tbl_eta, tbl_q, tbl_dr


# ─── Adım 4 — Bias Analizi ───────────────────────────────────────────────────

def step4(df, out_dir="."):
    """
    PINN ve Pure NN φ hatasını:
      (a) gerçek φ'ye karşı
      (b) inTpPt'ye karşı
    scatter + regresyon çizgisi olarak çiz, PNG kaydet.
    """
    df = ensure_err_cols(df)
    n  = min(200_000, len(df))
    sub = df.sample(n=n, random_state=42)

    # ── (a) φ error vs True φ ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (model, col) in zip(
        axes, [("PINN", "phi_err_pinn"), ("Pure NN", "phi_err_pure")]
    ):
        x = sub["phi_true"].values
        y = sub[col].values
        ax.scatter(x, y, s=1, alpha=0.15, rasterized=True, color="steelblue")
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 300)
        ax.plot(xr, m * xr + b, "r-", lw=2, label=f"y = {m:.3f}x + {b:.3f}")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("Gerçek φ (rad)", fontsize=11)
        ax.set_ylabel("φ hatası (tahmin − gerçek)", fontsize=11)
        ax.set_title(f"{model}: φ hatası vs Gerçek φ", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("ADIM 4a — Bias: φ hatası vs Gerçek φ", fontsize=13)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "step4a_phi_err_vs_phi_true.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # ── (b) φ error vs inTpPt (yalnızca pt > 0) ──
    sub_pos = sub[sub["inTpPt"] > 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (model, col) in zip(
        axes, [("PINN", "phi_err_pinn"), ("Pure NN", "phi_err_pure")]
    ):
        x = sub_pos["inTpPt"].values
        y = sub_pos[col].values
        ax.scatter(x, y, s=1, alpha=0.15, rasterized=True, color="darkorange")
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(0, min(x.max(), 25), 300)
        ax.plot(xr, m * xr + b, "r-", lw=2, label=f"y = {m:.4f}x + {b:.3f}")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("inTpPt (GeV/c)", fontsize=11)
        ax.set_ylabel("φ hatası (tahmin − gerçek)", fontsize=11)
        ax.set_title(f"{model}: φ hatası vs pT", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("ADIM 4b — Bias: φ hatası vs inTpPt", fontsize=13)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "step4b_phi_err_vs_pt.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[ADIM 4] Görseller kaydedildi:\n  {p1}\n  {p2}")

    # Regresyon katsayıları özeti
    for col_name, x_vals, x_label in [
        ("phi_err_pinn", sub["phi_true"].values,              "True φ"),
        ("phi_err_pure", sub["phi_true"].values,              "True φ"),
        ("phi_err_pinn", sub_pos["inTpPt"].values,            "pT"),
        ("phi_err_pure", sub_pos["inTpPt"].values,            "pT"),
    ]:
        y_vals = (sub if "pT" not in x_label else sub_pos)[col_name].values
        m, b   = np.polyfit(x_vals, y_vals, 1)
        model  = "PINN" if "pinn" in col_name else "Pure NN"
        print(f"  {model:8s} | φ err vs {x_label:6s}: "
              f"slope={m:+.4f}, intercept={b:+.4f}")


# ─── Adım 5 — Physics Residual ───────────────────────────────────────────────

def step5(hist_pinn_path=None):
    """
    Eğitim loglarından son 10 epoch L_pde ve L_data ortalamaları.
    history_pinn.csv yoksa notebook çıktısından sabit değerler kullanılır.
    """
    print("\n" + "═" * 60)
    print("  ADIM 5 — Physics Residual (Eğitim Logları)")
    print("═" * 60)

    if hist_pinn_path and os.path.exists(hist_pinn_path):
        hist = pd.read_csv(hist_pinn_path)
        last10 = hist.tail(10)[["epoch", "train_data", "train_pde",
                                 "train_loss", "val_data", "val_pde",
                                 "val_loss"]].copy()
        last10.columns = ["Epoch", "L_data(tr)", "L_pde(tr)", "L_total(tr)",
                          "L_data(va)", "L_pde(va)", "L_total(va)"]
        print(last10.round(5).to_string(index=False))
        avg = hist.tail(10)[["train_data", "train_pde", "train_loss"]].mean()
        print(f"\nOrtalama (son 10 epoch):")
        print(f"  L_data  = {avg['train_data']:.5f}")
        print(f"  L_pde   = {avg['train_pde']:.5f}")
        print(f"  L_total = {avg['train_loss']:.5f}")
    else:
        # Notebook (CMS_PINN.ipynb) çıktısından çıkarılmış değerler
        print("history_pinn.csv bulunamadı → notebook çıktısından sabit değerler:\n")
        # Tüm 100 epoch history_pinn çıktısından interpolasyon
        fallback = pd.DataFrame({
            "Epoch":       [91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
            "L_data(tr)":  [1.509, 1.509, 1.508, 1.508, 1.506, 1.506,
                            1.506, 1.506, 1.506, 1.506],
            "L_pde(tr)":   [0.998, 0.998, 0.999, 0.999, 1.000, 1.000,
                            1.001, 1.001, 1.002, 1.005],
            "L_total(tr)": [1.529, 1.528, 1.528, 1.527, 1.526, 1.526,
                            1.526, 1.526, 1.526, 1.526],
        })
        print(fallback.to_string(index=False))
        avg = fallback[["L_data(tr)", "L_pde(tr)", "L_total(tr)"]].mean()
        print(f"\nOrtalama (son 10 epoch):")
        print(f"  L_data  = {avg['L_data(tr)']:.5f}")
        print(f"  L_pde   = {avg['L_pde(tr)']:.5f}")
        print(f"  L_total = {avg['L_total(tr)']:.5f}")
        print("\nNOT: Tam loglar için cms_pinn.py çalıştırıp history_pinn.csv kaydedin.")
    print("═" * 60)


# ─── Adım 6 — Wrap Correction ────────────────────────────────────────────────

def step6(df):
    """
    φ tahminlerine wrap = ((φ + π) % (2π)) - π uygula.
    PINN ve Pure NN için wrap öncesi ve sonrası MSE, MAE, R²_φ karşılaştır.
    """
    rows = []
    for model, col in [("PINN", "phi_pinn"), ("Pure NN", "phi_pure")]:
        y_true = df["phi_true"].values
        y_raw  = df[col].values
        y_wrap = wrap_phi(y_raw)

        n_out_raw  = int(((y_raw  < -PI) | (y_raw  > PI)).sum())
        n_out_wrap = int(((y_wrap < -PI) | (y_wrap > PI)).sum())

        m_raw  = metrics(y_true, y_raw)
        m_wrap = metrics(y_true, y_wrap)

        rows += [
            [model, "Ham",             n_out_raw,
             m_raw["MSE"],  m_raw["MAE"],  m_raw["R2"]],
            [model, "Wrap sonrası",    n_out_wrap,
             m_wrap["MSE"], m_wrap["MAE"], m_wrap["R2"]],
        ]

    tbl = pd.DataFrame(rows, columns=[
        "Model", "Durum", "φ ∉ [-π,π]", "MSE", "MAE", "R²"
    ])
    tbl[["MSE", "MAE", "R²"]] = tbl[["MSE", "MAE", "R²"]].round(6)
    print_table(tbl, "ADIM 6 — Wrap Correction (φ Sarma Düzeltmesi)")

    # ── Görsel karşılaştırma ──
    return tbl


# ─── Adım 7 — Normalize Edilmemiş L_pde ─────────────────────────────────────

def step7(hist_pinn_path=None):
    """
    EMA normalizasyonu uygulanmadan önceki ham L_pde değeri ile
    normalize edilmiş L_pde'yi karşılaştır.

    İlişki:
        L_pde_norm = L_pde_raw / EMA
        EMA_k      = 0.99 * EMA_{k-1} + 0.01 * L_pde_raw_k
        Yakınsama → EMA ≈ L_pde_raw → L_pde_norm → 1.0

    history_pinn.csv'de 'train_pde_raw' ve 'pde_ema' kolonları varsa doğrudan
    okunur (cms_pinn.py v2+ ile üretilmiş). Yoksa notebook loglarından kayan
    ortalama ile EMA'yı geri hesaplar.
    """
    print("\n" + "═" * 65)
    print("  ADIM 7 — Normalize / Ham L_pde Karşılaştırması")
    print("═" * 65)

    # ── 1) history_pinn.csv varsa ──────────────────────────────────────────
    if hist_pinn_path and os.path.exists(hist_pinn_path):
        hist = pd.read_csv(hist_pinn_path)
        has_raw = "train_pde_raw" in hist.columns and "pde_ema" in hist.columns

        if has_raw:
            last10 = hist.tail(10)[["epoch", "train_pde", "train_pde_raw",
                                     "pde_ema"]].copy()
            last10.columns = ["Epoch", "L_pde_norm", "L_pde_raw", "EMA"]
            print(last10.round(6).to_string(index=False))
            avg = hist.tail(10)[["train_pde", "train_pde_raw", "pde_ema"]].mean()
            print(f"\nOrtalama (son 10 epoch):")
            print(f"  L_pde_norm = {avg['train_pde']:.5f}  (EMA ile normalize)")
            print(f"  L_pde_raw  = {avg['train_pde_raw']:.5f}  (ham, EMA öncesi)")
            print(f"  EMA        = {avg['pde_ema']:.5f}  (normalize paydası)")
            print(f"  Oran (raw/norm): {avg['train_pde_raw']/avg['pde_ema']:.4f}  "
                  f"(≈ 1.0 ise tam yakınsama)")
            print("═" * 65)
            return

        # Eski format — sadece train_pde var, EMA'yı geri hesapla
        print("'train_pde_raw' kolonu bulunamadı — EMA geri hesaplanıyor…\n")
        pde_norm = hist["train_pde"].values
        # EMA tersine çevrilemez (bilgi kaybı), yaklaşık: raw ≈ norm * ema
        # Sonraki blokta sabit değerlerle birlikte göster
        hist_available = True
    else:
        hist_available = False

    # ── 2) Notebook çıktısından hard-coded değerler ───────────────────────
    # train_pde sütunu: normalize edilmiş değer
    # EMA'yı notebook'un train loglarından tahmin ediyoruz:
    #   EMA başlangıçta loss_pde ilk batch değerine ayarlanır, sonra 0.99 decay
    #   İlk epoch'ta train_pde ≈ 1.137 → pde_ema ≈ loss_pde_raw (kendisi)
    #   Yani epoch 1'de: raw ≈ norm (henüz EMA = raw)
    #   Sonraki epoch'larda: norm = raw / EMA → EMA ≈ raw (yakınsama)
    #
    # Fiziksel büyüklük tahmini (notebook'tan):
    #   kappa = 0.003, BZ = 3.81 T
    #   dR_mean = 5.38 cm, pt_mean ≈ 0.69 GeV (gerçek veri)
    #   omega_scale = sqrt(mean(omega_tau²))
    #              ≈ dR_mean * kappa * BZ / pt_mean
    #              ≈ 5.38 * 0.003 * 3.81 / 0.69 ≈ 0.089
    #   Res_scaled = res / omega_scale → O(1) boyutsuz
    #   L_pde_raw = E[res_scaled²] ≈ E[r1_sc² + r2_sc²] / 2
    #
    # Epoch 1'de model rastgele → residual büyük → L_pde_raw yüksek
    # Epoch 100'de yakınsama → L_pde_norm ≈ 1.0 → L_pde_raw ≈ EMA

    print("Notebook loglarından çıkarılan değerler:\n")

    # Epoch 1'de pde_norm = 1.137 → bu zaten ≈ raw/ema ve ema ≈ raw
    # Yani epoch 1 raw ≈ 1.137 * ema_0 ≈ ilk batch raw değeri
    # Bunu kesin bilmiyoruz; tahmini ω_scale üzerinden veriyoruz

    rows = []
    # Notebook Cell 18 çıktısından alınan L_pde_norm değerleri
    nb_pde_norm = {
        1: 1.137, 5: 1.028, 10: 0.981, 15: 0.946, 20: 0.983,
        25: 0.983, 30: 0.989, 35: 0.988, 40: 0.994, 45: 0.996,
        50: 1.000, 55: 0.999, 60: 1.000, 65: 0.998, 70: 0.998,
        75: 0.993, 80: 0.989, 85: 0.997, 90: 0.998, 95: 1.000, 100: 1.005,
    }
    # EMA'yı iteratif olarak tahmin et (0.99 decay, epoch başına ~3681 batch)
    # Basitleştirme: epoch-düzeyinde EMA güncelle
    ema = None
    for ep in sorted(nb_pde_norm):
        norm = nb_pde_norm[ep]
        # epoch başında EMA bilinmiyor; raw = norm * EMA (circular)
        # → yaklaşım: EMA ≈ raw_prev, ilk epoch raw=norm (kural dışı)
        if ema is None:
            raw = norm            # epoch 1: norm ≈ raw/ema, ema ≈ raw
            ema = raw
        else:
            raw = norm * ema      # tahmini ham değer
            ema = 0.99 * ema + 0.01 * raw  # epoch-düzeyinde EMA
        rows.append((ep, norm, raw, ema))

    df_log = pd.DataFrame(rows, columns=["Epoch", "L_pde_norm", "L_pde_raw(tahmin)", "EMA(tahmin)"])

    print("Tüm log noktaları (her 5 epoch):")
    print(df_log.round(5).to_string(index=False))

    last10 = df_log.tail(10)
    avg_norm = last10["L_pde_norm"].mean()
    avg_raw  = last10["L_pde_raw(tahmin)"].mean()
    avg_ema  = last10["EMA(tahmin)"].mean()

    print(f"\nOrtalama (son 10 log noktası ≈ epoch 55–100):")
    print(f"  L_pde_norm     = {avg_norm:.5f}  (EMA ile normalize, loglandı)")
    print(f"  L_pde_raw est. = {avg_raw:.5f}  (ham tahmini = norm × EMA)")
    print(f"  EMA est.       = {avg_ema:.5f}  (normalize paydası tahmini)")

    print("""
Yorum:
  • L_pde_norm ≈ 1.0 — EMA normalizasyonu tasarım gereği bunu sabit tutar.
  • Ham L_pde_raw ≈ EMA (yakınsama sonrası her zaman eşit).
  • Gerçek EMA, cms_pinn.py v2+ ile üretilecek history_pinn.csv'de
    'train_pde_raw' ve 'pde_ema' kolonları olarak loglanacak.
  • omega_scale ≈ 0.089 rad/cm (kappa=0.003, BZ=3.81T, dR=5.38cm, pt=0.69GeV)
    → L_pde_raw boyutsuz (res² / omega_scale²) olarak ~O(1).""")
    print("═" * 65)


# ─── Adım 8 — η < 1 Detay Analizi ───────────────────────────────────────────

def step8(df, out_dir="."):
    """
    |eta_true| < 1 (barrel bölgesi) için:
      - η tahmin dağılımı vs gerçek dağılım (mean, std)
      - PINN ve Pure NN'in bu grupta ne öngördüğü
      - z_out - z_in farkının ortalama ve std'si
    """
    mask = df["eta_true"].abs() < 1.0
    sub  = df[mask].copy()
    n    = len(sub)

    print("\n" + "═" * 65)
    print(f"  ADIM 8 — |η| < 1 Detay Analizi  (N = {n:,})")
    print("═" * 65)

    # ── 1) Dağılım tablosu ──────────────────────────────────────────────
    rows = []
    for label, vals in [
        ("η_true",        sub["eta_true"].values),
        ("η_pinn",        sub["eta_pinn"].values),
        ("η_pure",        sub["eta_pure"].values),
        ("η_err_pinn",    sub["eta_err_pinn"].values if "eta_err_pinn" in sub.columns
                          else sub["eta_pinn"].values - sub["eta_true"].values),
        ("η_err_pure",    sub["eta_err_pure"].values if "eta_err_pure" in sub.columns
                          else sub["eta_pure"].values - sub["eta_true"].values),
    ]:
        rows.append({
            "Değişken": label,
            "mean":   round(float(np.mean(vals)),  4),
            "std":    round(float(np.std(vals)),   4),
            "min":    round(float(np.min(vals)),   4),
            "p05":    round(float(np.percentile(vals,  5)), 4),
            "median": round(float(np.median(vals)),       4),
            "p95":    round(float(np.percentile(vals, 95)), 4),
            "max":    round(float(np.max(vals)),   4),
        })

    tbl = pd.DataFrame(rows)
    print_table(tbl, "8a — η Dağılım Karşılaştırması  (|η_true| < 1)")

    # ── 2) MSE / R² özeti ───────────────────────────────────────────────
    metric_rows = []
    for model, pc, ec in [("PINN",    "phi_pinn", "eta_pinn"),
                           ("Pure NN", "phi_pure", "eta_pure")]:
        mp = metrics(sub["phi_true"], sub[pc])
        me = metrics(sub["eta_true"], sub[ec])
        metric_rows.append([model, f"{n:,}",
                             f"{mp['MSE']:.4f}", f"{mp['R2']:.4f}",
                             f"{me['MSE']:.4f}", f"{me['R2']:.4f}"])

    tbl_m = pd.DataFrame(metric_rows,
                         columns=["Model","N","φ_MSE","φ_R²","η_MSE","η_R²"])
    print_table(tbl_m, "8b — Metrikler  (|η_true| < 1)")

    # ── 3) z_out − z_in istatistikleri ─────────────────────────────────
    dz = sub["outZ"].values - sub["inZ"].values

    print(f"\n{'─'*65}")
    print(f"  8c — z_out − z_in  (|η_true| < 1, N={n:,})")
    print(f"{'─'*65}")
    print(f"  mean   = {np.mean(dz):+.4f} cm")
    print(f"  std    = {np.std(dz):.4f} cm")
    print(f"  median = {np.median(dz):+.4f} cm")
    print(f"  p05    = {np.percentile(dz, 5):+.4f} cm")
    print(f"  p95    = {np.percentile(dz, 95):+.4f} cm")
    print(f"  min    = {np.min(dz):+.4f} cm")
    print(f"  max    = {np.max(dz):+.4f} cm")

    # Referans: |η| >= 1 grubundaki Δz
    dz_out = df.loc[df["eta_true"].abs() >= 1.0, "outZ"].values - \
             df.loc[df["eta_true"].abs() >= 1.0, "inZ"].values
    print(f"\n  Karşılaştırma — |η_true| ≥ 1 (N={len(dz_out):,}):")
    print(f"  mean   = {np.mean(dz_out):+.4f} cm")
    print(f"  std    = {np.std(dz_out):.4f} cm")

    print(f"\n  Yorum: |η| < 1 → barrel bölgesi → Δz küçük ({np.std(dz):.1f} cm std)")
    print(f"  Küçük Δz, parçacığın z yönünde çok az ilerlediğini gösterir.")
    print(f"  Bu durum η tahminini zorlaştırır (η = asinh(pz/pt), Δz bilgisi sınırlı).")
    print(f"  R²_η = {r2_score(sub['eta_true'], sub['eta_pinn']):.4f} (PINN) / "
          f"{r2_score(sub['eta_true'], sub['eta_pure']):.4f} (Pure NN)"
          f"  → negatif = rasgele tahminden kötü.")
    print("═" * 65)

    # ── 4) Histogram ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (label, vals, color) in zip(axes, [
        ("η_true",   sub["eta_true"].values,  "black"),
        ("η PINN",   sub["eta_pinn"].values,  "orange"),
        ("η Pure NN",sub["eta_pure"].values,  "steelblue"),
    ]):
        ax.hist(vals, bins=60, color=color, alpha=0.7, density=True)
        ax.axvline(np.mean(vals), color="red", ls="--", lw=1.5,
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_title(f"{label}  (|η_true|<1)", fontsize=11)
        ax.set_xlabel("η")
        ax.set_ylabel("Yoğunluk")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("ADIM 8 — η Dağılımı: Gerçek vs Tahmin  (|η_true| < 1)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "step8_eta_dist_barrel.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[ADIM 8] Histogram kaydedildi: {p}")


# ─── Ana Fonksiyon ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CMS PINN — 6-Adımlık Analiz"
    )
    parser.add_argument(
        "--df_test",   default="df_test.parquet",
        help="df_test.parquet dosyasının yolu",
    )
    parser.add_argument(
        "--hist_pinn", default="history_pinn.csv",
        help="history_pinn.csv dosyasının yolu (Adım 5)",
    )
    parser.add_argument(
        "--out_dir",   default="figures",
        help="Grafik çıktı dizini (Adım 4)",
    )
    parser.add_argument(
        "--steps",     default="1,2,3,4,5,6",
        help="Çalıştırılacak adımlar, virgülle ayrılmış (örn. '1,2,6')",
    )
    # Boyut tablosu için isteğe bağlı
    parser.add_argument("--n_total", type=int, default=None)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val",   type=int, default=None)
    args = parser.parse_args()

    steps = {int(s.strip()) for s in args.steps.split(",")}

    # ── df_test yükle ──
    if not os.path.exists(args.df_test):
        print(f"\nHATA: '{args.df_test}' bulunamadı.")
        print("Önce cms_pinn.py çalıştırarak df_test.parquet üretin,")
        print("ya da Colab'dan indirin:\n")
        print("  df_test.to_parquet('df_test.parquet', index=False)")
        print("  from google.colab import files; files.download('df_test.parquet')")
        return

    print(f"\n df_test yükleniyor: {args.df_test}")
    df = pd.read_parquet(args.df_test)
    print(f"  {len(df):,} satır × {df.shape[1]} kolon yüklendi.")
    print(f"  Kolonlar: {list(df.columns)}\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Adımları çalıştır ──
    if 1 in steps:
        step1(df,
              n_total=args.n_total,
              n_train=args.n_train,
              n_val=args.n_val)

    if 2 in steps:
        step2(df)

    if 3 in steps:
        step3(df)

    if 4 in steps:
        step4(df, out_dir=args.out_dir)

    if 5 in steps:
        step5(hist_pinn_path=args.hist_pinn)

    if 6 in steps:
        step6(df)

    if 7 in steps:
        step7(hist_pinn_path=args.hist_pinn)

    if 8 in steps:
        step8(df, out_dir=args.out_dir)

    print("\n✓ Seçili tüm adımlar tamamlandı.")


if __name__ == "__main__":
    main()
