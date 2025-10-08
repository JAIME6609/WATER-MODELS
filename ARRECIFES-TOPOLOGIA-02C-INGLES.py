# -*- coding: utf-8 -*-
"""
Memory-efficient and dependency-robust pipeline:
- AE (1D vs 2D) on circular data (float32)
- Plots: latents, reconstructions/overlays, MSE histogram, training curves
- TDA with subsampling: diagrams/barcodes (H0, H1 [, optional H2]) and Betti curves
- Per-dimension Wasserstein distances
- Excel export (per-epoch losses, MSE metrics, distances, metadata)
- Automatic selection/installation of Excel engine: openpyxl/xlsxwriter

To avoid OOM/RAM issues:
- Subsampling for ripser (TDA_SUBSAMPLE)
- Optional H2 (ALLOW_H2=False by default)
- Figures using 'Agg' backend, figure closing and gc.collect()

!pip install -q tensorflow numpy scikit-learn matplotlib pandas openpyxl ripser persim
"""

# ===============================================================
# 0) Imports and utilities
# ===============================================================
import os, sys, subprocess, gc, importlib
import numpy as np
import pandas as pd

# Non-interactive backend to save memory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K

# Light reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def _ensure_package(pkg_name):
    """Try to import a package; if it fails, attempt installation and re-import."""
    try:
        importlib.import_module(pkg_name)
        return True
    except Exception:
        try:
            print(f"[Installing] {pkg_name} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            importlib.import_module(pkg_name)
            return True
        except Exception as e:
            print(f"[Warning] Could not install {pkg_name}: {e}")
            return False

# Attempt to ensure TDA libraries
HAS_RIPSER = _ensure_package("ripser")
HAS_PERSIM = _ensure_package("persim")

if HAS_RIPSER:
    from ripser import ripser
if HAS_PERSIM:
    from persim import wasserstein, plot_diagrams

# ===============================================================
# 1) General configuration (memory aware)
# ===============================================================
RANDOM_SEED   = 42
N_SAMPLES     = 1200           # Can be increased; TDA uses separate subsampling
NOISE_LEVEL   = 0.05
TEST_SIZE     = 0.2
EPOCHS        = 100
BATCH_SIZE    = 64

# --- Memory-efficient TDA config ---
TDA_SUBSAMPLE = 600            # max number of points for TDA (ripser); lower if RAM is limited
ALLOW_H2      = True           # True to compute H2 (more expensive)
TDA_MAXDIM    = 1 if not ALLOW_H2 else 2  # H0..H1 by default; optional H2

# Outputs
OUT_DIR       = Path("experiment_outputs")
IMG_DIR       = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
EXCEL_PATH    = OUT_DIR / "topo_ae_results.xlsx"

# Figure style
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 130,
    "axes.grid": True
})

# ===============================================================
# 2) Synthetic data (ring) and split
# ===============================================================
def generate_circle_data(n_samples=1200, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    theta = (2 * np.pi * rng.random(n_samples)).astype(np.float32)
    x = (np.cos(theta) + noise * rng.standard_normal(n_samples)).astype(np.float32)
    y = (np.sin(theta) + noise * rng.standard_normal(n_samples)).astype(np.float32)
    return np.vstack((x, y)).T.astype(np.float32)

X = generate_circle_data(N_SAMPLES, NOISE_LEVEL, RANDOM_SEED).astype(np.float32)
X_train, X_test = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# ===============================================================
# 3) AE models: 1D and 2D (float32)
# ===============================================================
input_dim = 2

# Classic AE 1D
latent_dim_classic = 1
inp_c = layers.Input(shape=(input_dim,), dtype="float32")
hc    = layers.Dense(32, activation="relu")(inp_c)
hc    = layers.Dense(32, activation="relu")(hc)
zc    = layers.Dense(latent_dim_classic, name="latent_1d")(hc)

zin_c = layers.Input(shape=(latent_dim_classic,), dtype="float32")
uc    = layers.Dense(32, activation="relu")(zin_c)
uc    = layers.Dense(32, activation="relu")(uc)
out_c = layers.Dense(input_dim)(uc)

encoder_classic = Model(inp_c, zc, name="encoder_classic_1d")
decoder_classic = Model(zin_c, out_c, name="decoder_classic_1d")
xhat_c = decoder_classic(encoder_classic(inp_c))
autoencoder_classic = Model(inp_c, xhat_c, name="autoencoder_classic_1d")
autoencoder_classic.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

hist_c = autoencoder_classic.fit(
    X_train, X_train,
    validation_data=(X_test, X_test),
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0
)

print(f"[Classic AE 1D] Final loss (train): {hist_c.history['loss'][-1]:.6f} | "
      f"Final loss (val): {hist_c.history['val_loss'][-1]:.6f}")

Z_classic     = encoder_classic.predict(X, verbose=0).astype(np.float32)
X_rec_classic = autoencoder_classic.predict(X, verbose=0).astype(np.float32)

# Free some TF intermediate memory
K.clear_session()
gc.collect()

# AE 2D
latent_dim_2d = 2
inp2 = layers.Input(shape=(input_dim,), dtype="float32")
h2   = layers.Dense(32, activation="relu")(inp2)
h2   = layers.Dense(32, activation="relu")(h2)
z2   = layers.Dense(latent_dim_2d, name="latent_2d")(h2)

zin2 = layers.Input(shape=(latent_dim_2d,), dtype="float32")
u2   = layers.Dense(32, activation="relu")(zin2)
u2   = layers.Dense(32, activation="relu")(u2)
out2 = layers.Dense(input_dim)(u2)

encoder_2d = Model(inp2, z2, name="encoder_2d")
decoder_2d = Model(zin2, out2, name="decoder_2d")
xhat2 = decoder_2d(encoder_2d(inp2))
autoencoder_2d = Model(inp2, xhat2, name="autoencoder_2d")
autoencoder_2d.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

hist_2d = autoencoder_2d.fit(
    X_train, X_train,
    validation_data=(X_test, X_test),
    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0
)

print(f"[AE 2D]       Final loss (train): {hist_2d.history['loss'][-1]:.6f} | "
      f"Final loss (val): {hist_2d.history['val_loss'][-1]:.6f}")

Z_2d     = encoder_2d.predict(X, verbose=0).astype(np.float32)
X_rec_2d = autoencoder_2d.predict(X, verbose=0).astype(np.float32)

gc.collect()

# ===============================================================
# 4) Per-point MSE metrics and statistics (float32)
# ===============================================================
mse_pt_classic = np.mean((X - X_rec_classic)**2, axis=1).astype(np.float32)
mse_pt_2d      = np.mean((X - X_rec_2d     )**2, axis=1).astype(np.float32)

print(f"[Per-point MSE] Classic AE 1D -> mean: {mse_pt_classic.mean():.6f} | "
      f"median: {np.median(mse_pt_classic):.6f}")
print(f"[Per-point MSE] AE 2D        -> mean: {mse_pt_2d.mean():.6f} | "
      f"median: {np.median(mse_pt_2d):.6f}")

gc.collect()

# ===============================================================
# 5) Base visualizations (lightweight figures + close/cleanup)
# ===============================================================
# (A) Latents
fig = plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
Zc = np.ravel(Z_classic)
plt.scatter(Zc, np.zeros_like(Zc), s=5)
plt.title("Classic AE latent (1D)")
plt.yticks([]); plt.grid(True, ls=":")

plt.subplot(1,3,2)
plt.scatter(Z_2d[:,0], Z_2d[:,1], s=5)
plt.title("AE latent (2D)")
plt.axis("equal"); plt.grid(True, ls=":")

plt.subplot(1,3,3)
plt.scatter(Z_2d[:,0], Z_2d[:,1], s=5, label="AE 2D")
plt.scatter(Zc, np.zeros_like(Zc), s=5, alpha=0.5, label="AE 1D (y=0)")
plt.title("Latent comparison")
plt.grid(True, ls=":"); plt.legend()

plt.tight_layout()
plt.savefig(IMG_DIR / "latents_comparison.png")
plt.close(fig)
gc.collect()

# (B) Reconstructions and overlays (subsample for plotting)
rng = np.random.default_rng(123)
idx = rng.choice(X.shape[0], size=int(min(500, X.shape[0])), replace=False)

fig = plt.figure(figsize=(14,8))
ax1 = plt.subplot(2,3,1)
ax1.scatter(X[idx,0], X[idx,1], s=6, label="Original")
ax1.set_title("Original"); ax1.set_aspect("equal"); ax1.legend(fontsize=8)

ax2 = plt.subplot(2,3,2)
ax2.scatter(X_rec_classic[idx,0], X_rec_classic[idx,1], s=6, label="Reconstr. Classic AE 1D")
ax2.set_title("Classic AE reconstruction (1D)"); ax2.set_aspect("equal"); ax2.legend(fontsize=8)

ax3 = plt.subplot(2,3,3)
ax3.scatter(X_rec_2d[idx,0], X_rec_2d[idx,1], s=6, label="Reconstr. AE 2D")
ax3.set_title("AE reconstruction (2D)"); ax3.set_aspect("equal"); ax3.legend(fontsize=8)

ax4 = plt.subplot(2,3,4)
ax4.scatter(X[idx,0], X[idx,1], s=6, label="Original")
ax4.scatter(X_rec_classic[idx,0], X_rec_classic[idx,1], s=6, alpha=0.6, label="AE 1D")
ax4.set_title("Overlay: Original vs AE 1D"); ax4.set_aspect("equal"); ax4.legend(fontsize=8)

ax5 = plt.subplot(2,3,5)
ax5.scatter(X[idx,0], X[idx,1], s=6, label="Original")
ax5.scatter(X_rec_2d[idx,0], X_rec_2d[idx,1], s=6, alpha=0.6, label="AE 2D")
ax5.set_title("Overlay: Original vs AE 2D"); ax5.set_aspect("equal"); ax5.legend(fontsize=8)

ax6 = plt.subplot(2,3,6)
bins = 30
ax6.hist(mse_pt_classic, bins=bins, alpha=0.7, label="AE 1D")
ax6.hist(mse_pt_2d,      bins=bins, alpha=0.7, label="AE 2D")
ax6.set_title("Per-point MSE distribution")
ax6.set_xlabel("MSE"); ax6.set_ylabel("Frequency"); ax6.legend(fontsize=8)

plt.suptitle("Reconstructions and error comparison", y=0.98, fontsize=13)
plt.tight_layout()
plt.savefig(IMG_DIR / "reconstructions_overlays_hist.png")
plt.close(fig)
gc.collect()

# (C) Training curves (MSE)
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist_c.history["loss"], label="train 1D")
plt.plot(hist_c.history["val_loss"], label="val 1D")
plt.title("Classic AE training (1D)")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True, ls=":")

plt.subplot(1,2,2)
plt.plot(hist_2d.history["loss"], label="train 2D")
plt.plot(hist_2d.history["val_loss"], label="val 2D")
plt.title("AE training (2D)")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.grid(True, ls=":")

plt.tight_layout()
plt.savefig(IMG_DIR / "training_curves.png")
plt.close(fig)
gc.collect()

# ===============================================================
# 6) Memory-efficient TDA: Diagrams/barcodes and Betti curves
# ===============================================================
def _clean_dgm(D):
    if D is None or (isinstance(D, np.ndarray) and D.size == 0):
        return np.zeros((0, 2), dtype=np.float32)
    D = np.asarray(D, dtype=np.float32)
    mask = np.isfinite(D).all(axis=1)
    D = D[mask]
    if D.ndim != 2 or D.shape[1] != 2:
        return np.zeros((0, 2), dtype=np.float32)
    return D

def compute_persistence_subsample(X_data, maxdim=1, n_perm=400):
    """
    Persistent homology with ripser using subsampling (n_perm) to avoid OOM.
    """
    if X_data.shape[0] > n_perm:
        rng = np.random.default_rng(7)
        sel = rng.choice(X_data.shape[0], size=n_perm, replace=False)
        X_use = X_data[sel].astype(np.float32, copy=False)
    else:
        X_use = X_data.astype(np.float32, copy=False)
    res = ripser(X_use, maxdim=maxdim, n_perm=X_use.shape[0])
    dgms = res["dgms"]
    dgms_clean = [ _clean_dgm(d) for d in dgms ]
    return dgms_clean

def plot_persistence_diagrams(dgms, title, savepath):
    fig = plt.figure(figsize=(6,5))
    plot_diagrams(dgms, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)

def plot_barcodes(dgms, title, savepath, max_bars=60):
    dims = len(dgms)
    fig, axes = plt.subplots(1, dims, figsize=(4*dims, 5), sharey=True)
    if dims == 1:
        axes = [axes]
    for dim, ax in enumerate(axes):
        D = dgms[dim]
        ax.set_title(f"Barcodes H{dim}")
        ax.set_xlabel("ε"); ax.set_yticks([])
        if D.size == 0:
            continue
        deads = D[:,1][np.isfinite(D[:,1])]
        cutoff = np.percentile(deads, 99) if deads.size > 0 else (D[:,0].max() if D[:,0].size>0 else 1.0)
        Kbars = min(len(D), max_bars)
        sel = np.argsort(-(D[:,1] - D[:,0]))[:Kbars]
        Dsel = D[sel]
        for i in range(Kbars):
            b, d = Dsel[i]
            d_plot = min(d, cutoff)
            ax.hlines(y=Kbars-1-i, xmin=b, xmax=d_plot, color="black", linewidth=2)
    fig.suptitle(title, y=0.98)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)

def betti_curve_from_dgm(dgm, eps_grid):
    if dgm is None or dgm.size == 0:
        return np.zeros_like(eps_grid, dtype=np.int32)
    births = dgm[:,0].astype(np.float32)
    deaths = dgm[:,1].astype(np.float32)
    big = eps_grid.max() + (eps_grid[1]-eps_grid[0] if eps_grid.size>1 else np.float32(1.0))
    deaths = np.where(np.isfinite(deaths), deaths, big).astype(np.float32)
    betti = np.zeros_like(eps_grid, dtype=np.int32)
    for i, eps in enumerate(eps_grid):
        betti[i] = int(np.sum((births <= eps) & (eps < deaths)))
    return betti

def plot_betti_curves(dgms_list, labels, dims, eps_grid, prefix_title, savepath):
    fig = plt.figure(figsize=(6,4.5))
    for dgms, lab in zip(dgms_list, labels):
        for dim in dims:
            if dim < len(dgms):
                bc = betti_curve_from_dgm(dgms[dim], eps_grid)
                plt.plot(eps_grid, bc, label=f"{lab} (H{dim})")
    plt.title(prefix_title)
    plt.xlabel("ε"); plt.ylabel("β")
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)

dgms_orig = dgms_ae1d = dgms_ae2d = None
dims_for_plots = [0,1] if TDA_MAXDIM == 1 else [0,1,2]

if HAS_RIPSER and HAS_PERSIM:
    # Subsampling to avoid OOM in TDA
    dgms_orig = compute_persistence_subsample(X, maxdim=TDA_MAXDIM, n_perm=TDA_SUBSAMPLE)
    dgms_ae1d = compute_persistence_subsample(X_rec_classic, maxdim=TDA_MAXDIM, n_perm=TDA_SUBSAMPLE)
    dgms_ae2d = compute_persistence_subsample(X_rec_2d, maxdim=TDA_MAXDIM, n_perm=TDA_SUBSAMPLE)

    # Persistence diagrams
    plot_persistence_diagrams(dgms_orig, "Persistence diagrams (Original)", IMG_DIR / "diagrams_original.png")
    plot_persistence_diagrams(dgms_ae1d, "Persistence diagrams (AE 1D)",    IMG_DIR / "diagrams_ae1d.png")
    plot_persistence_diagrams(dgms_ae2d, "Persistence diagrams (AE 2D)",    IMG_DIR / "diagrams_ae2d.png")

    # Barcodes
    plot_barcodes(dgms_orig, "Barcodes (Original)", IMG_DIR / "barcodes_original.png")
    plot_barcodes(dgms_ae1d, "Barcodes (AE 1D)",    IMG_DIR / "barcodes_ae1d.png")
    plot_barcodes(dgms_ae2d, "Barcodes (AE 2D)",    IMG_DIR / "barcodes_ae2d.png")

    # Betti curves (grid from min/max)
    def _minmax_across(dgms_all):
        births, deaths = [], []
        for dgms in dgms_all:
            for dgm in dgms:
                if dgm.size > 0:
                    births.append(np.min(dgm[:,0]))
                    finite_deaths = dgm[:,1][np.isfinite(dgm[:,1])]
                    deaths.append(np.max(finite_deaths) if finite_deaths.size>0 else np.max(dgm[:,0]))
        bmin = float(np.min(births)) if births else 0.0
        dmax = float(np.max(deaths)) if deaths else 1.0
        return bmin, dmax

    bmin, dmax = _minmax_across([dgms_orig, dgms_ae1d, dgms_ae2d])
    eps_grid = np.linspace(bmin, dmax, 160).astype(np.float32)

    plot_betti_curves(
        [dgms_orig, dgms_ae1d, dgms_ae2d],
        ["Original", "AE 1D", "AE 2D"],
        dims_for_plots,
        eps_grid,
        prefix_title="Betti curves (subsampled)",
        savepath=IMG_DIR / "betti_curves.png"
    )

gc.collect()

# ===============================================================
# 7) Wasserstein distances (per available dimension)
# ===============================================================
def wasserstein_per_dim(dgms_ref, dgms_hat, dims=(0,1)):
    dists = {}
    for d in dims:
        if (dgms_ref is None) or (dgms_hat is None) or d >= len(dgms_ref) or d >= len(dgms_hat):
            dists[f"H{d}"] = np.nan
            continue
        A = _clean_dgm(dgms_ref[d]); B = _clean_dgm(dgms_hat[d])
        dists[f"H{d}"] = (float(wasserstein(A, B)) if (A.size>0 and B.size>0) else np.nan)
    return dists

wasserstein_ae1d = {}
wasserstein_ae2d = {}

if HAS_RIPSER and HAS_PERSIM:
    dims_tuple = tuple(dims_for_plots)
    wasserstein_ae1d = wasserstein_per_dim(dgms_orig, dgms_ae1d, dims=dims_tuple)
    wasserstein_ae2d = wasserstein_per_dim(dgms_orig, dgms_ae2d, dims=dims_tuple)
    print("[Wasserstein] Original vs AE 1D:", wasserstein_ae1d)
    print("[Wasserstein] Original vs AE 2D:", wasserstein_ae2d)
else:
    print("TDA skipped: 'ripser' and/or 'persim' are not available.")

gc.collect()

# ===============================================================
# 8) Robust Excel engine selection and export
# ===============================================================
def _pick_excel_engine():
    """
    Select 'openpyxl' or 'xlsxwriter'. Try to import; if not, install.
    Return the engine name as string.
    """
    # Prefer openpyxl due to typical availability
    try:
        importlib.import_module("openpyxl")
        return "openpyxl"
    except Exception:
        pass
    try:
        importlib.import_module("xlsxwriter")
        return "xlsxwriter"
    except Exception:
        pass

    # Try installing openpyxl
    if _ensure_package("openpyxl"):
        return "openpyxl"
    # Try installing xlsxwriter
    if _ensure_package("xlsxwriter"):
        return "xlsxwriter"

    raise RuntimeError(
        "It was not possible to make an Excel engine ('openpyxl' or 'xlsxwriter') available. "
        "Please install one of them manually."
    )

EXCEL_ENGINE = _pick_excel_engine()
print(f"[Excel] Using engine: {EXCEL_ENGINE}")

def export_to_excel(
    excel_path,
    engine,
    hist_c, hist_2d,
    mse_pt_classic, mse_pt_2d,
    wasserstein_ae1d, wasserstein_ae2d,
    dims_for_plots
):
    with pd.ExcelWriter(excel_path, engine=engine) as writer:
        # Per-epoch losses (AE 1D)
        df_loss_1d = pd.DataFrame({
            "epoch": np.arange(1, len(hist_c.history["loss"])+1, dtype=np.int32),
            "loss_train": np.asarray(hist_c.history["loss"], dtype=np.float32),
            "loss_val":   np.asarray(hist_c.history["val_loss"], dtype=np.float32)
        })
        df_loss_1d.to_excel(writer, sheet_name="losses_ae1d", index=False)

        # Per-epoch losses (AE 2D)
        df_loss_2d = pd.DataFrame({
            "epoch": np.arange(1, len(hist_2d.history["loss"])+1, dtype=np.int32),
            "loss_train": np.asarray(hist_2d.history["loss"], dtype=np.float32),
            "loss_val":   np.asarray(hist_2d.history["val_loss"], dtype=np.float32)
        })
        df_loss_2d.to_excel(writer, sheet_name="losses_ae2d", index=False)

        # Per-point MSE statistics
        df_stats = pd.DataFrame({
            "model": ["AE_1D", "AE_2D"],
            "mse_mean": [float(mse_pt_classic.mean()), float(mse_pt_2d.mean())],
            "mse_median": [float(np.median(mse_pt_classic)), float(np.median(mse_pt_2d))],
            "mse_std": [float(mse_pt_classic.std(ddof=1)), float(mse_pt_2d.std(ddof=1))]
        })
        df_stats.to_excel(writer, sheet_name="mse_stats", index=False)

        # Per-point MSE (useful for percentiles/fine analysis)
        pd.DataFrame({"mse_point": mse_pt_classic.astype(np.float32)}).to_excel(
            writer, sheet_name="mse_point_ae1d", index=False
        )
        pd.DataFrame({"mse_point": mse_pt_2d.astype(np.float32)}).to_excel(
            writer, sheet_name="mse_point_ae2d", index=False
        )

        # Per-dimension Wasserstein distances
        if len(wasserstein_ae1d)>0 or len(wasserstein_ae2d)>0:
            dims_txt = [f"H{d}" for d in dims_for_plots]
            df_wass = pd.DataFrame({
                "dimension": dims_txt,
                "wasserstein_vs_AE1D": [wasserstein_ae1d.get(k, np.nan) for k in dims_txt],
                "wasserstein_vs_AE2D": [wasserstein_ae2d.get(k, np.nan) for k in dims_txt],
            })
            df_wass.to_excel(writer, sheet_name="wasserstein", index=False)

        # Metadata
        df_meta = pd.DataFrame({
            "parameter": ["N_SAMPLES", "NOISE_LEVEL", "TEST_SIZE", "EPOCHS", "BATCH_SIZE",
                          "TDA_SUBSAMPLE", "TDA_MAXDIM", "ALLOW_H2", "EXCEL_ENGINE"],
            "value":     [N_SAMPLES,   NOISE_LEVEL,   TEST_SIZE,   EPOCHS,   BATCH_SIZE,
                          TDA_SUBSAMPLE, TDA_MAXDIM,  ALLOW_H2,    EXCEL_ENGINE]
        })
        df_meta.to_excel(writer, sheet_name="metadata", index=False)

export_to_excel(
    EXCEL_PATH,
    EXCEL_ENGINE,
    hist_c, hist_2d,
    mse_pt_classic, mse_pt_2d,
    wasserstein_ae1d, wasserstein_ae2d,
    dims_for_plots
)

print("\n=== Process completed (memory-efficient and robust Excel) ===")
print(f"- Figures: {IMG_DIR.resolve()}")
print(f"- Excel:   {EXCEL_PATH.resolve()}")
if not (HAS_RIPSER and HAS_PERSIM):
    print("Note: Install 'ripser' and 'persim' to enable diagrams/barcodes/Betti curves and Wasserstein distances.")

