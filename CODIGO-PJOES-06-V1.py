#!/usr/bin/env python3
"""
Topological deep learning pipeline for forecasting water-quality deterioration.

This script implements the methodology described in Sections 1-4 of the manuscript:
1. It reads a real CSV dataset or generates a synthetic smart-city water-quality benchmark.
2. It preprocesses mixed physicochemical and microbiological time series.
3. It converts rolling windows into persistence images using delay embeddings and persistent homology.
4. It trains three predictive models:
      - Raw-only temporal model
      - Topology-only model
      - Fusion model (raw + persistence images)
5. It exports subsection-ready tables and figures into:
      results/5.1
      results/5.2
      results/5.3

Important design decisions:
- No GUDHI is used.
- Topological libraries include Ripser, Persim, kmapper/KeplerMapper, and TopoNetX.
- The code is intentionally verbose and highly commented to facilitate adaptation.
- A synthetic benchmark is included so the pipeline can run end-to-end even without field data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ripser import ripser
from persim import PersistenceImager
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

# Both package names point to the same library ecosystem.
import kmapper as km
from kmapper import KeplerMapper
import toponetx as tnx


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """Operational thresholds used to build the deterioration score.

    Positive variables deteriorate when they become too large.
    Negative variables deteriorate when they become too small.
    """

    positive_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "turbidity_ntu": 8.0,
            "conductivity_uscm": 650.0,
            "ammonia_mgL": 0.80,
            "total_coliform_cfu100mL": 200.0,
            "e_coli_cfu100mL": 100.0,
        }
    )
    negative_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "dissolved_oxygen_mgL": 5.0,
        }
    )
    positive_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "turbidity_ntu": 0.90,
            "conductivity_uscm": 0.80,
            "ammonia_mgL": 1.00,
            "total_coliform_cfu100mL": 1.10,
            "e_coli_cfu100mL": 1.30,
        }
    )
    negative_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "dissolved_oxygen_mgL": 1.20,
        }
    )
    event_threshold: float = 2.50


@dataclass
class Config:
    """Master configuration for the experiment pipeline."""

    seed: int = 42
    n_steps: int = 1800
    freq: str = "30min"
    window: int = 32
    stride: int = 4
    lead: int = 6
    train_frac: float = 0.70
    val_frac: float = 0.15

    # Time-delay embedding and persistence image settings
    embedding_dim: int = 3
    delay: int = 2
    birth_range: Tuple[float, float] = (0.0, 3.2)
    pers_range: Tuple[float, float] = (0.0, 3.2)
    pixel_size: float = 0.2
    persistence_power: float = 1.0
    sigma: float = 0.15

    # Training settings
    batch_size: int = 64
    epochs: int = 18
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_cls: float = 1.0
    lambda_reg: float = 0.06
    patience: int = 5

    # Input columns for the synthetic benchmark and most expected real datasets
    feature_vars: List[str] = field(
        default_factory=lambda: [
            "dissolved_oxygen_mgL",
            "turbidity_ntu",
            "conductivity_uscm",
            "ph",
            "ammonia_mgL",
            "total_coliform_cfu100mL",
            "e_coli_cfu100mL",
            "rainfall_mm",
            "flow_m3s",
            "temperature_c",
        ]
    )
    topo_vars: List[str] = field(
        default_factory=lambda: [
            "dissolved_oxygen_mgL",
            "turbidity_ntu",
            "conductivity_uscm",
            "ammonia_mgL",
            "total_coliform_cfu100mL",
            "e_coli_cfu100mL",
        ]
    )
    skewed_log_vars: List[str] = field(
        default_factory=lambda: [
            "total_coliform_cfu100mL",
            "e_coli_cfu100mL",
        ]
    )
    contextual_vars: List[str] = field(
        default_factory=lambda: [
            "rainfall_mm",
            "flow_m3s",
            "temperature_c",
        ]
    )
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: Path) -> None:
    """Create a directory and all parents if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)



def save_json(obj: Dict, path: Path) -> None:
    """Write a JSON file with indentation for readability."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)



def save_dataframe_table(df: pd.DataFrame, path_csv: Path, path_png: Path, title: str, font_size: int = 9) -> None:
    """Save a table as both CSV and PNG.

    The PNG version is convenient for direct insertion into a manuscript.
    The CSV version preserves machine-readable values for later verification.
    """
    df.to_csv(path_csv, index=False)

    fig_w = max(8.0, 1.3 * df.shape[1])
    fig_h = max(2.4, 0.42 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.25)
    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)



def try_import_torch_directml():
    """Attempt to import torch-directml only when it is needed.

    The import is optional because many CUDA-focused environments do not ship with
    torch-directml installed. Returning ``None`` keeps the pipeline usable on
    Linux, macOS, and standard CUDA installations.
    """
    try:
        import torch_directml  # type: ignore
        return torch_directml
    except Exception:
        return None



def cuda_architecture_supported(device_index: int = 0) -> Tuple[bool, Dict[str, Any]]:
    """Check whether the installed PyTorch binary supports the detected CUDA arch.

    This is particularly important for NVIDIA Blackwell laptop GPUs such as the
    RTX 5050 Laptop GPU. Older PyTorch wheels can *detect* the GPU and still fail
    at execution time because they do not contain kernels for ``sm_120``.
    """
    info: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "device_index": device_index,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "supported_arches": [],
        "required_arch": None,
        "device_name": None,
        "reason": None,
    }

    if not torch.cuda.is_available():
        info["reason"] = "torch.cuda.is_available() returned False."
        return False, info

    try:
        info["device_name"] = torch.cuda.get_device_name(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        info["required_arch"] = f"sm_{major}{minor}"
        try:
            info["supported_arches"] = [str(x) for x in torch.cuda.get_arch_list()]
        except Exception as exc:
            info["supported_arches"] = []
            info["arch_list_error"] = f"{type(exc).__name__}: {exc}"

        supported = (
            info["required_arch"] in info["supported_arches"]
            or f"compute_{major}{minor}" in info["supported_arches"]
        )
        if supported:
            info["reason"] = "The detected CUDA architecture is present in the installed PyTorch binary."
        else:
            supported_arches = ", ".join(info["supported_arches"]) if info["supported_arches"] else "none reported"
            info["reason"] = (
                f"Detected GPU architecture {info['required_arch']}, but the installed PyTorch binary reports "
                f"support for: {supported_arches}."
            )
        return supported, info
    except Exception as exc:
        info["reason"] = f"Failed to query CUDA capability: {type(exc).__name__}: {exc}"
        return False, info



def smoke_test_backend(device: Any) -> Tuple[bool, Optional[str]]:
    """Run a tiny tensor operation on the selected backend.

    A smoke test is safer than relying on ``torch.cuda.is_available()`` alone,
    because some unsupported CUDA installations fail only after the first kernel
    launch.
    """
    try:
        a = torch.randn(8, 8).to(device)
        b = torch.randn(8, 8).to(device)
        c = (a + b).sum()
        _ = float(c.detach().cpu().item())
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"



def select_compute_device(requested: str = "auto") -> Tuple[Any, Dict[str, Any]]:
    """Select CUDA, DirectML, or CPU with explicit compatibility checks.

    Selection order for ``auto``:
        1. CUDA, but only when the installed PyTorch wheel supports the detected GPU.
        2. DirectML on Windows if ``torch-directml`` is installed and passes a smoke test.
        3. CPU as a safe fallback.
    """
    requested = str(requested).lower().strip()
    if requested not in {"auto", "cpu", "cuda", "directml"}:
        raise ValueError(f"Unsupported device request: {requested}")

    info: Dict[str, Any] = {
        "requested": requested,
        "selected_backend": None,
        "device_repr": None,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "notes": [],
    }

    def finalize(device: Any, backend: str, note: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        info["selected_backend"] = backend
        info["device_repr"] = str(device)
        if note:
            info["notes"].append(note)
        return device, info

    if requested == "cpu":
        return finalize(torch.device("cpu"), "cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        supported, cuda_info = cuda_architecture_supported()
        info["cuda"] = cuda_info
        if not supported:
            raise RuntimeError(
                "CUDA was requested, but the installed PyTorch binary does not support the detected GPU "
                f"architecture ({cuda_info.get('required_arch')}). Install a CUDA 12.8 build of PyTorch "
                "2.7.0 or newer, or run with --device directml / --device auto."
            )
        device = torch.device("cuda")
        ok, err = smoke_test_backend(device)
        if not ok:
            raise RuntimeError(f"CUDA was detected, but the backend smoke test failed: {err}")
        return finalize(device, "cuda")

    if requested == "directml":
        torch_directml = try_import_torch_directml()
        if torch_directml is None:
            raise RuntimeError(
                "DirectML was requested, but torch-directml is not installed in the active environment."
            )
        device = torch_directml.device()
        ok, err = smoke_test_backend(device)
        if not ok:
            raise RuntimeError(f"DirectML was requested, but the backend smoke test failed: {err}")
        info["directml"] = {"available": True}
        return finalize(device, "directml")

    # Auto mode
    supported, cuda_info = cuda_architecture_supported()
    info["cuda"] = cuda_info
    if torch.cuda.is_available() and supported:
        device = torch.device("cuda")
        ok, err = smoke_test_backend(device)
        if ok:
            return finalize(device, "cuda")
        info["notes"].append(f"CUDA smoke test failed: {err}")
    elif torch.cuda.is_available() and not supported:
        info["notes"].append(
            "CUDA is visible, but the installed PyTorch binary does not contain kernels for the detected architecture."
        )

    torch_directml = try_import_torch_directml()
    if torch_directml is not None:
        device = torch_directml.device()
        ok, err = smoke_test_backend(device)
        if ok:
            info["directml"] = {"available": True}
            return finalize(
                device,
                "directml",
                note="DirectML was selected because CUDA was unavailable, unsupported, or failed its smoke test.",
            )
        info["notes"].append(f"DirectML smoke test failed: {err}")

    return finalize(
        torch.device("cpu"),
        "cpu",
        note="CPU was selected because neither CUDA nor DirectML was ready for execution.",
    )



def move_to_device(x: torch.Tensor, device: Any) -> torch.Tensor:
    """Move a tensor to any supported backend device."""
    return x.to(device)





def sanitize_binary_targets_and_scores(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove non-finite classification entries and coerce targets to binary integers."""
    y_true = np.asarray(y_true).reshape(-1)
    score = np.asarray(score).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(score)
    if mask.sum() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    y = np.rint(y_true[mask]).astype(np.int64)
    s = np.clip(score[mask].astype(np.float32), 0.0, 1.0)
    return y, s



def sanitize_regression_targets_and_predictions(y_true: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove non-finite regression entries before metric computation."""
    y_true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(pred)
    if mask.sum() == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return y_true[mask].astype(np.float32), pred[mask].astype(np.float32)



def has_two_classes(y_true: np.ndarray) -> bool:
    """Return True only when both binary classes are present."""
    y = np.asarray(y_true).reshape(-1)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return False
    return len(np.unique(np.rint(y).astype(int))) >= 2



def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Compute AUROC safely even when one class is absent."""
    y, s = sanitize_binary_targets_and_scores(y_true, score)
    if len(y) == 0 or not has_two_classes(y):
        return float("nan")
    return float(roc_auc_score(y, s))



def safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Compute AUPRC safely even when one class is absent."""
    y, s = sanitize_binary_targets_and_scores(y_true, score)
    if len(y) == 0 or not has_two_classes(y):
        return float("nan")
    return float(average_precision_score(y, s))



def safe_balanced_accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Compute balanced accuracy without invalid one-class values."""
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(pred).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    y = np.rint(y[mask]).astype(int)
    p = np.rint(p[mask]).astype(int)
    if not has_two_classes(y):
        return float("nan")
    return float(balanced_accuracy_score(y, p))



def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Choose a classification threshold by maximizing F1 on validation data."""
    y, p = sanitize_binary_targets_and_scores(y_true, prob)
    if len(y) == 0 or not has_two_classes(y):
        return 0.50
    best_thr = 0.50
    best_f1 = -1.0
    for thr in np.linspace(0.10, 0.90, 161):
        pred = (p >= thr).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr



def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute stable binary-classification metrics after removing invalid entries."""
    y, p = sanitize_binary_targets_and_scores(y_true, prob)
    if len(y) == 0:
        return {
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
            "F1": float("nan"),
            "Precision": float("nan"),
            "Recall": float("nan"),
            "BalancedAcc": float("nan"),
        }
    pred = (p >= threshold).astype(int)
    return {
        "AUROC": safe_auc(y, p),
        "AUPRC": safe_auprc(y, p),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "BalancedAcc": safe_balanced_accuracy(y, pred),
    }



def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics after removing NaN and infinite values."""
    y, p = sanitize_regression_targets_and_predictions(y_true, pred)
    if len(y) == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
        }
    mae = float(mean_absolute_error(y, p))
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    r2 = float("nan") if len(y) < 2 else float(r2_score(y, p))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# -----------------------------------------------------------------------------
# Synthetic benchmark generator
# -----------------------------------------------------------------------------


def generate_synthetic_water_quality(cfg: Config) -> pd.DataFrame:
    """Create a synthetic smart-city water-quality benchmark.

    The generator intentionally encodes several urban contamination regimes:
    - runoff-driven turbidity pulses
    - sewer overflow episodes
    - industrial-conductivity anomalies
    - low-flow microbial regrowth

    This benchmark is not a substitute for field deployment.
    Its role is to make the full topological pipeline runnable without external data.
    """
    rng = np.random.default_rng(cfg.seed)
    idx = pd.date_range("2025-01-01", periods=cfg.n_steps, freq=cfg.freq)
    t = np.arange(cfg.n_steps)

    rainfall = np.zeros(cfg.n_steps, dtype=float)
    i = 0
    while i < cfg.n_steps:
        if rng.random() < 0.055:
            duration = int(rng.integers(2, 16))
            intensity = rng.gamma(shape=2.3, scale=2.6)
            decay = np.exp(-np.linspace(0, 2.2, duration))
            upper = min(cfg.n_steps, i + duration)
            rainfall[i:upper] += intensity * decay[: upper - i]
            i += duration
        else:
            i += 1
    rainfall += rng.gamma(0.35, 0.12, cfg.n_steps)
    rainfall = np.clip(rainfall, 0.0, None)

    seasonal = 22.0 + 4.0 * np.sin(2 * np.pi * t / (24 * 12))
    diurnal = 1.0 * np.sin(2 * np.pi * t / 48)
    temperature = seasonal + diurnal + rng.normal(0, 0.35, cfg.n_steps)

    runoff_kernel = np.exp(-np.arange(0, 24) / 4.8)
    runoff_memory = np.convolve(rainfall, runoff_kernel, mode="full")[: cfg.n_steps]
    flow = 1.8 + 0.24 * runoff_memory + 0.08 * np.sin(2 * np.pi * t / 48 + 0.5) + rng.normal(0, 0.05, cfg.n_steps)
    flow = np.clip(flow, 0.2, None)

    # Regime coding used to create hidden contamination states.
    regime = np.zeros(cfg.n_steps, dtype=int)
    remaining = 0
    current = 0
    for i in range(cfg.n_steps):
        if remaining > 0:
            regime[i] = current
            remaining -= 1
            if current in (1, 2) and rainfall[i] > 2.0 and rng.random() < 0.25:
                remaining += 1
            continue

        wet = runoff_memory[i]
        if rainfall[i] > 3.2 and wet > 9.0 and rng.random() < 0.60:
            current = 2  # sewer overflow
            remaining = int(rng.integers(5, 14))
        elif rainfall[i] > 1.4 and rng.random() < 0.50:
            current = 1  # runoff
            remaining = int(rng.integers(4, 11))
        elif rng.random() < 0.016:
            current = 3  # industrial anomaly
            remaining = int(rng.integers(3, 8))
        elif temperature[i] > 24 and flow[i] < 2.3 and rng.random() < 0.03:
            current = 4  # microbial regrowth
            remaining = int(rng.integers(5, 11))
        else:
            current = 0
            remaining = 0
        regime[i] = current

    runoff = (regime == 1).astype(float)
    overflow = (regime == 2).astype(float)
    industrial = (regime == 3).astype(float)
    regrowth = (regime == 4).astype(float)

    turbidity = np.zeros(cfg.n_steps)
    dissolved_oxygen = np.zeros(cfg.n_steps)
    conductivity = np.zeros(cfg.n_steps)
    ph = np.zeros(cfg.n_steps)
    ammonia = np.zeros(cfg.n_steps)
    total_coliform = np.zeros(cfg.n_steps)
    e_coli = np.zeros(cfg.n_steps)

    turbidity[0] = 2.5
    dissolved_oxygen[0] = 7.9
    conductivity[0] = 425.0
    ph[0] = 7.25
    ammonia[0] = 0.12
    total_coliform[0] = 55.0
    e_coli[0] = 18.0

    for i in range(1, cfg.n_steps):
        turbidity[i] = (
            0.72 * turbidity[i - 1]
            + 0.35 * rainfall[i]
            + 8.0 * overflow[i]
            + 3.1 * runoff[i]
            + 1.8 * industrial[i]
            + rng.normal(0, 0.65)
        )
        turbidity[i] = max(0.2, turbidity[i])

        conductivity[i] = (
            0.86 * conductivity[i - 1]
            + 58
            + 24 * runoff[i]
            + 90 * industrial[i]
            + 62 * overflow[i]
            - 7 * rainfall[i]
            + rng.normal(0, 11)
        )
        conductivity[i] = np.clip(conductivity[i], 180, None)

        ammonia[i] = (
            0.74 * ammonia[i - 1]
            + 0.05
            + 0.26 * overflow[i]
            + 0.18 * industrial[i]
            + 0.08 * runoff[i]
            + 0.03 * max(temperature[i] - 23, 0)
            + rng.normal(0, 0.035)
        )
        ammonia[i] = np.clip(ammonia[i], 0.01, None)

        ph[i] = (
            0.81 * ph[i - 1]
            + 1.42
            - 0.08 * industrial[i]
            - 0.04 * overflow[i]
            - 0.02 * rainfall[i]
            + 0.03 * np.sin(2 * np.pi * i / 96)
            + rng.normal(0, 0.03)
        )
        ph[i] = np.clip(ph[i], 6.2, 8.6)

        dissolved_oxygen[i] = (
            0.82 * dissolved_oxygen[i - 1]
            + 1.28
            - 0.05 * (temperature[i] - 22)
            - 0.018 * turbidity[i]
            - 0.33 * ammonia[i]
            - 0.72 * overflow[i]
            - 0.23 * regrowth[i]
            + 0.08 * np.sin(2 * np.pi * i / 48 + 0.9)
            + rng.normal(0, 0.10)
        )
        dissolved_oxygen[i] = np.clip(dissolved_oxygen[i], 1.5, 11.0)

        total_coliform[i] = (
            0.78 * total_coliform[i - 1]
            + 10
            + 100 * overflow[i]
            + 58 * regrowth[i]
            + 20 * runoff[i]
            + 0.8 * max(temperature[i] - 23, 0)
            + rng.normal(0, 7)
        )
        total_coliform[i] = np.clip(total_coliform[i], 5, None)

        e_coli[i] = (
            0.72 * e_coli[i - 1]
            + 3
            + 0.33 * total_coliform[i]
            + 45 * overflow[i]
            + 25 * regrowth[i]
            + rng.normal(0, 4.5)
        )
        e_coli[i] = np.clip(e_coli[i], 1, None)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "rainfall_mm": rainfall,
            "flow_m3s": flow,
            "temperature_c": temperature,
            "turbidity_ntu": turbidity,
            "dissolved_oxygen_mgL": dissolved_oxygen,
            "conductivity_uscm": conductivity,
            "ph": ph,
            "ammonia_mgL": ammonia,
            "total_coliform_cfu100mL": total_coliform,
            "e_coli_cfu100mL": e_coli,
            "regime": regime,
        }
    )

    regime_map = {
        0: "normal",
        1: "runoff",
        2: "sewer_overflow",
        3: "industrial_discharge",
        4: "microbial_regrowth",
    }
    df["regime_name"] = df["regime"].map(regime_map)

    # Introduce realistic missingness.
    for col in ["total_coliform_cfu100mL", "e_coli_cfu100mL"]:
        keep = np.zeros(cfg.n_steps, dtype=bool)
        keep[::4] = True
        keep &= ~(rng.random(cfg.n_steps) < 0.05)
        df.loc[~keep, col] = np.nan

    for col in [
        "dissolved_oxygen_mgL",
        "turbidity_ntu",
        "conductivity_uscm",
        "ph",
        "ammonia_mgL",
        "temperature_c",
    ]:
        miss = rng.random(cfg.n_steps) < 0.015
        for start in np.where(miss)[0]:
            if rng.random() < 0.35:
                duration = int(rng.integers(2, 5))
                miss[start : start + duration] = True
        df.loc[miss, col] = np.nan

    return df


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------


def load_input_dataset(cfg: Config, data_csv: Optional[str]) -> pd.DataFrame:
    """Load a user dataset when provided; otherwise create the synthetic benchmark."""
    if data_csv is None:
        print("[INFO] No data CSV was provided. Generating the synthetic benchmark...")
        return generate_synthetic_water_quality(cfg)

    path = Path(data_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("The input CSV must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df




def compute_deterioration_targets(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute the continuous deterioration score and binary warning label.

    Missing values in threshold-governed variables are handled neutrally rather than
    propagating NaNs into the targets. A missing value is forward-filled when past
    information exists; otherwise it falls back to the threshold itself, which
    contributes zero deterioration for that variable.
    """
    def neutral_series(values: pd.Series, neutral_value: float) -> np.ndarray:
        series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series.notna().any():
            series = series.ffill().fillna(neutral_value)
        else:
            series = pd.Series(np.full(len(series), neutral_value), index=series.index, dtype=float)
        return series.to_numpy(dtype=float)

    positive_sum = np.zeros(len(df), dtype=float)
    negative_sum = np.zeros(len(df), dtype=float)

    for col, threshold in cfg.threshold_config.positive_thresholds.items():
        if col not in df.columns:
            warnings.warn(f"Positive threshold column '{col}' is missing and will be ignored.")
            continue
        weight = cfg.threshold_config.positive_weights.get(col, 1.0)
        values = neutral_series(df[col], threshold)
        contribution = weight * np.maximum(0.0, (values - threshold) / max(threshold, 1e-6))
        positive_sum += np.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)

    for col, threshold in cfg.threshold_config.negative_thresholds.items():
        if col not in df.columns:
            warnings.warn(f"Negative threshold column '{col}' is missing and will be ignored.")
            continue
        weight = cfg.threshold_config.negative_weights.get(col, 1.0)
        values = neutral_series(df[col], threshold)
        contribution = weight * np.maximum(0.0, (threshold - values) / max(threshold, 1e-6))
        negative_sum += np.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)

    deterioration_score = positive_sum + negative_sum
    rolling_term = pd.Series(deterioration_score).rolling(6, min_periods=1).mean().to_numpy()
    deterioration_score = np.nan_to_num(
        deterioration_score + 0.35 * rolling_term,
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )

    df = df.copy()
    df["deterioration_score"] = deterioration_score.astype(float)
    df["event_label"] = (df["deterioration_score"] > cfg.threshold_config.event_threshold).astype(int)
    return df




def attach_missing_masks(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Create explicit missingness masks for all feature variables that exist."""
    df = df.copy()
    for col in cfg.feature_vars:
        if col in df.columns:
            df[f"{col}_mask"] = df[col].isna().astype(int)
    return df




def preprocess_time_series(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], List[str]]:
    """Perform split-aware robust preprocessing."""
    df = df.copy()
    df = attach_missing_masks(df, cfg)

    active_feature_vars = [col for col in cfg.feature_vars if col in df.columns]
    if len(active_feature_vars) == 0:
        raise ValueError("No expected feature columns were found in the dataset.")

    for col in active_feature_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    for col in cfg.skewed_log_vars:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0.0))

    n_train = int(len(df) * cfg.train_frac)
    train_block = df.iloc[:n_train][active_feature_vars].replace([np.inf, -np.inf], np.nan)

    medians_series = train_block.median(skipna=True).fillna(0.0)
    q75 = train_block.quantile(0.75)
    q25 = train_block.quantile(0.25)
    iqr_series = (q75 - q25).replace([np.inf, -np.inf], np.nan).fillna(1.0).replace(0.0, 1.0)

    medians = {k: float(v) for k, v in medians_series.to_dict().items()}
    iqrs = {k: float(v) for k, v in iqr_series.to_dict().items()}

    for col in active_feature_vars:
        df[col] = df[col].ffill()
        df[col] = df[col].fillna(medians[col])
        df[col] = (df[col] - medians[col]) / (iqrs[col] + 1e-6)
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df, medians, iqrs, active_feature_vars



# -----------------------------------------------------------------------------
# Topological feature generation
# -----------------------------------------------------------------------------


def delay_embedding(series: Sequence[float], dim: int, tau: int) -> np.ndarray:
    """Create a Takens-style delay embedding from a one-dimensional series."""
    arr = np.asarray(series, dtype=float)
    n = len(arr) - (dim - 1) * tau
    if n <= 1:
        return np.zeros((0, dim), dtype=float)
    return np.column_stack([arr[i : i + n] for i in range(0, dim * tau, tau)])


class PITransformer:
    """Small wrapper around Persim's PersistenceImager.

    The wrapper ensures consistent output size and safe handling of empty diagrams.
    """

    def __init__(self, cfg: Config):
        def weight_fn(birth: float, persistence: float, n: float = 1.0) -> float:
            return persistence ** n

        self.height = int(round((cfg.pers_range[1] - cfg.pers_range[0]) / cfg.pixel_size))
        self.width = int(round((cfg.birth_range[1] - cfg.birth_range[0]) / cfg.pixel_size))
        self.imager = PersistenceImager(
            birth_range=cfg.birth_range,
            pers_range=cfg.pers_range,
            pixel_size=cfg.pixel_size,
            weight=weight_fn,
            weight_params={"n": cfg.persistence_power},
            kernel_params={"sigma": [[cfg.sigma, 0.0], [0.0, cfg.sigma]]},
        )

    def transform(self, diagram: np.ndarray) -> np.ndarray:
        if diagram is None or len(diagram) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)

        arr = np.asarray(diagram, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return np.zeros((self.height, self.width), dtype=np.float32)

        arr = arr[np.isfinite(arr[:, 1])]
        arr = arr[arr[:, 1] > arr[:, 0]]
        if len(arr) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)

        bp = arr.copy()
        bp[:, 1] = bp[:, 1] - bp[:, 0]
        bp = bp[(bp[:, 0] >= 0) & (bp[:, 1] >= 0)]
        if len(bp) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)

        image = self.imager.transform(bp, skew=False).astype(np.float32)
        out = np.zeros((self.height, self.width), dtype=np.float32)
        h = min(self.height, image.shape[0])
        w = min(self.width, image.shape[1])
        out[:h, :w] = image[:h, :w]
        return out




def build_window_bank(df: pd.DataFrame, cfg: Config, active_feature_vars: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """Create raw windows, persistence-image tensors, and metadata."""
    pitrans = PITransformer(cfg)

    active_topo_vars = [col for col in cfg.topo_vars if col in df.columns]
    if len(active_topo_vars) == 0:
        raise ValueError("No topological feature columns were found in the dataset.")

    active_mask_vars = [f"{col}_mask" for col in active_feature_vars if f"{col}_mask" in df.columns]
    raw_columns = active_feature_vars + active_mask_vars

    raw_list: List[np.ndarray] = []
    topo_list: List[np.ndarray] = []
    meta_rows: List[Dict] = []

    for end in range(cfg.window - 1, len(df) - cfg.lead, cfg.stride):
        start = end - cfg.window + 1
        target_idx = end + cfg.lead
        window_df = df.iloc[start : end + 1]

        raw_seq = window_df[raw_columns].values.astype(np.float32)
        raw_seq = np.nan_to_num(raw_seq, nan=0.0, posinf=0.0, neginf=0.0)

        channels = []
        for col in active_topo_vars:
            series = np.asarray(window_df[col].values, dtype=float)
            series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
            emb = delay_embedding(series, dim=cfg.embedding_dim, tau=cfg.delay)
            if emb.shape[0] < 3:
                diagrams = [np.empty((0, 2)), np.empty((0, 2))]
            else:
                diagrams = ripser(emb, maxdim=1)["dgms"][:2]
            channels.append(pitrans.transform(diagrams[0]))
            channels.append(pitrans.transform(diagrams[1]))

        topo_tensor = np.stack(channels, axis=0).astype(np.float32)
        topo_tensor = np.nan_to_num(topo_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        target_score = float(df.iloc[target_idx]["deterioration_score"])
        if not np.isfinite(target_score):
            target_score = 0.0

        target_label_raw = df.iloc[target_idx]["event_label"]
        if pd.isna(target_label_raw):
            target_label = int(target_score > cfg.threshold_config.event_threshold)
        else:
            target_label = int(target_label_raw)

        raw_list.append(raw_seq)
        topo_list.append(topo_tensor)
        meta_rows.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "target_idx": int(target_idx),
                "window_start": df.iloc[start]["timestamp"],
                "window_end": df.iloc[end]["timestamp"],
                "target_time": df.iloc[target_idx]["timestamp"],
                "target_score": target_score,
                "target_label": target_label,
                "target_regime": str(df.iloc[target_idx].get("regime_name", "unknown")),
            }
        )

    if len(raw_list) == 0:
        raise ValueError(
            "No rolling windows were generated. Provide a longer dataset or reduce the values of window, lead, and stride."
        )

    raw_bank = np.stack(raw_list).astype(np.float32)
    topo_bank = np.stack(topo_list).astype(np.float32)
    raw_bank = np.nan_to_num(raw_bank, nan=0.0, posinf=0.0, neginf=0.0)
    topo_bank = np.nan_to_num(topo_bank, nan=0.0, posinf=0.0, neginf=0.0)

    meta_df = pd.DataFrame(meta_rows)
    meta_df["target_score"] = pd.to_numeric(meta_df["target_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill_value = float(meta_df["target_score"].median()) if meta_df["target_score"].notna().any() else 0.0
    meta_df["target_score"] = meta_df["target_score"].fillna(fill_value)
    meta_df["target_label"] = pd.to_numeric(meta_df["target_label"], errors="coerce").fillna(0).astype(int)

    return raw_bank, topo_bank, meta_df, active_mask_vars



# -----------------------------------------------------------------------------
# Optional topological summaries for descriptive analysis
# -----------------------------------------------------------------------------


def build_sensor_complex_report(df: pd.DataFrame, topo_vars: List[str]) -> pd.DataFrame:
    """Create a small simplicial-complex summary from variable correlations.

    This is not used for model training. It provides an additional topological
    descriptive object for Section 5.1 using TopoNetX.
    """
    active = [col for col in topo_vars if col in df.columns]
    corr = df[active].corr().abs()
    sc = tnx.SimplicialComplex()

    for v in active:
        sc.add_simplex([v])

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            if corr.iloc[i, j] >= 0.35:
                sc.add_simplex([active[i], active[j]])

    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            for k in range(j + 1, len(active)):
                if (
                    corr.iloc[i, j] >= 0.35
                    and corr.iloc[i, k] >= 0.35
                    and corr.iloc[j, k] >= 0.35
                ):
                    sc.add_simplex([active[i], active[j], active[k]])

    rows = []
    for rank in range(1, 4):
        cells = list(sc.skeleton(rank - 1))
        rows.append({"Complex rank": rank - 1, "Number of simplices": len(cells)})

    rows.append({"Complex rank": "Maximal", "Number of simplices": len(sc.simplices)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# PyTorch dataset and models
# -----------------------------------------------------------------------------



class WindowDataset(Dataset):
    """Dataset wrapper for multimodal windows."""

    def __init__(self, raw: np.ndarray, topo: np.ndarray, y_reg: np.ndarray, y_cls: np.ndarray):
        raw = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        topo = np.nan_to_num(np.asarray(topo, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_reg = np.nan_to_num(np.asarray(y_reg, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_cls = np.clip(np.nan_to_num(np.asarray(y_cls, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        self.raw = torch.tensor(raw, dtype=torch.float32)
        self.topo = torch.tensor(topo, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32).unsqueeze(-1)
        self.y_cls = torch.tensor(y_cls, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.y_cls)

    def __getitem__(self, idx: int):
        return self.raw[idx], self.topo[idx], self.y_reg[idx], self.y_cls[idx]



class RawTemporalEncoder(nn.Module):
    """A compact 1D CNN for raw multivariate windows."""

    def __init__(self, input_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(48, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))


class TopologyEncoder(nn.Module):
    """A compact 2D CNN for persistence-image tensors."""

    def __init__(self, in_channels: int, hidden_dim: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHead(nn.Module):
    """Shared hidden layer with regression and classification outputs."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        return self.reg_head(h), self.cls_head(h)


class RawOnlyNet(nn.Module):
    """Baseline model that uses only raw temporal windows."""

    def __init__(self, raw_dim: int):
        super().__init__()
        self.raw_enc = RawTemporalEncoder(raw_dim, hidden_dim=48)
        self.head = MultiTaskHead(48, hidden_dim=48)

    def forward(self, raw: torch.Tensor, topo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.raw_enc(raw))


class TopologyOnlyNet(nn.Module):
    """Baseline model that uses only persistence-image tensors."""

    def __init__(self, topo_channels: int):
        super().__init__()
        self.topo_enc = TopologyEncoder(topo_channels, hidden_dim=48)
        self.head = MultiTaskHead(48, hidden_dim=48)

    def forward(self, raw: torch.Tensor, topo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.topo_enc(topo))


class FusionNet(nn.Module):
    """Proposed multimodal model combining raw and topological branches."""

    def __init__(self, raw_dim: int, topo_channels: int):
        super().__init__()
        self.raw_enc = RawTemporalEncoder(raw_dim, hidden_dim=48)
        self.topo_enc = TopologyEncoder(topo_channels, hidden_dim=48)
        self.fusion_proj = nn.Linear(96, 96)
        self.fusion_gate = nn.Linear(96, 96)
        self.head = MultiTaskHead(96, hidden_dim=64)

    def forward(self, raw: torch.Tensor, topo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_h = self.raw_enc(raw)
        topo_h = self.topo_enc(topo)
        fused = torch.cat([raw_h, topo_h], dim=1)
        activated = torch.relu(self.fusion_proj(fused))
        gate = torch.sigmoid(self.fusion_gate(fused))
        hidden = activated * gate
        return self.head(hidden)


# -----------------------------------------------------------------------------
# Training and evaluation helpers
# -----------------------------------------------------------------------------



def collect_predictions(model: nn.Module, loader: DataLoader, device: Any) -> Dict[str, np.ndarray]:
    """Collect regression predictions, logits, and targets from a data loader."""
    model.eval()
    reg_list, logit_list, y_reg_list, y_cls_list = [], [], [], []
    with torch.no_grad():
        for raw, topo, y_reg, y_cls in loader:
            raw = move_to_device(raw, device)
            topo = move_to_device(topo, device)
            reg_pred, cls_logit = model(raw, topo)
            reg_list.append(reg_pred.detach().cpu().numpy().ravel())
            logit_list.append(cls_logit.detach().cpu().numpy().ravel())
            y_reg_list.append(y_reg.numpy().ravel())
            y_cls_list.append(y_cls.numpy().ravel())
    if len(reg_list) == 0:
        empty = np.array([], dtype=np.float32)
        return {"reg_pred": empty, "prob": empty, "y_reg": empty, "y_cls": empty}
    reg_pred = np.nan_to_num(np.concatenate(reg_list), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    logits = np.nan_to_num(np.concatenate(logit_list), nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
    y_reg = np.nan_to_num(np.concatenate(y_reg_list), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    y_cls = np.clip(np.nan_to_num(np.concatenate(y_cls_list), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    return {"reg_pred": reg_pred, "prob": prob, "y_reg": y_reg, "y_cls": y_cls}





def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Config,
    device: Any,
) -> Tuple[nn.Module, Dict]:
    """Train one model with early stopping and return metrics and predictions."""
    if len(train_loader.dataset) == 0:
        raise ValueError("The training split is empty. Increase the dataset length or adjust the split fractions.")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_targets = [batch[3].numpy().ravel() for batch in train_loader]
    if len(train_targets) == 0:
        raise ValueError("No training batches were produced by the DataLoader.")
    train_y = np.clip(np.nan_to_num(np.concatenate(train_targets, axis=0), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    pos_count = float(train_y.sum())
    neg_count = float(len(train_y) - pos_count)
    pos_weight_value = neg_count / max(pos_count, 1.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device))
    mse = nn.MSELoss()

    best_state = None
    best_score = -np.inf
    wait = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        for raw, topo, y_reg, y_cls in train_loader:
            raw = move_to_device(raw, device)
            topo = move_to_device(topo, device)
            y_reg = move_to_device(y_reg, device)
            y_cls = move_to_device(y_cls, device)

            optimizer.zero_grad()
            reg_pred, cls_logit = model(raw, topo)
            loss = cfg.lambda_cls * bce(cls_logit, y_cls) + cfg.lambda_reg * mse(reg_pred, y_reg)
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    "A non-finite training loss was encountered. Check the input data, targets, and backend installation."
                )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(y_cls)

        val_pred = collect_predictions(model, val_loader, device)
        threshold = best_threshold(val_pred["y_cls"], val_pred["prob"])
        val_cls = classification_metrics(val_pred["y_cls"], val_pred["prob"], threshold)
        score = 0.70 * np.nan_to_num(val_cls["AUROC"], nan=0.0) + 0.30 * np.nan_to_num(val_cls["F1"], nan=0.0)

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(len(train_loader.dataset), 1),
                "val_auroc": val_cls["AUROC"],
                "val_auprc": val_cls["AUPRC"],
                "val_f1": val_cls["F1"],
                "threshold": threshold,
            }
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a best model state.")

    model.load_state_dict(best_state)
    train_pred = collect_predictions(model, train_loader, device)
    val_pred = collect_predictions(model, val_loader, device)
    test_pred = collect_predictions(model, test_loader, device)
    threshold = best_threshold(val_pred["y_cls"], val_pred["prob"])

    result = {
        "threshold": threshold,
        "history": history,
        "train_class": classification_metrics(train_pred["y_cls"], train_pred["prob"], threshold),
        "val_class": classification_metrics(val_pred["y_cls"], val_pred["prob"], threshold),
        "test_class": classification_metrics(test_pred["y_cls"], test_pred["prob"], threshold),
        "train_reg": regression_metrics(train_pred["y_reg"], train_pred["reg_pred"]),
        "val_reg": regression_metrics(val_pred["y_reg"], val_pred["reg_pred"]),
        "test_reg": regression_metrics(test_pred["y_reg"], test_pred["reg_pred"]),
        "predictions": {
            "train": train_pred,
            "val": val_pred,
            "test": test_pred,
        },
    }
    return model, result



# -----------------------------------------------------------------------------
# Results generation
# -----------------------------------------------------------------------------


def select_representative_index(
    meta: pd.DataFrame,
    label: Optional[int] = None,
    quantile: float = 0.50,
) -> Optional[int]:
    """Select a representative window index using target-score proximity.

    Parameters
    ----------
    meta:
        Window-level metadata containing at least ``target_score`` and, when class
        filtering is requested, ``target_label``.
    label:
        Optional binary class constraint. When provided, the representative window
        is selected only from that class.
    quantile:
        Quantile of ``target_score`` used as the reference severity level.

    Returns
    -------
    Optional[int]
        Index into ``raw_bank``/``topo_bank`` and ``meta_df``. ``None`` is returned
        when no candidate exists.
    """
    if meta is None or len(meta) == 0:
        return None

    pool = meta
    if label is not None:
        pool = pool.loc[pool["target_label"] == label]
    if pool.empty:
        return None

    reference = float(pool["target_score"].quantile(quantile))
    ranking = (pool["target_score"] - reference).abs().sort_values()
    if len(ranking) == 0:
        return None
    return int(ranking.index[0])



def choose_descriptive_windows(meta_df: pd.DataFrame, test_mask: np.ndarray) -> Tuple[int, int, str, str]:
    """Choose robust representative windows for Section 5.1 visualization.

    The original implementation assumed that the test split always contained both a
    negative and a positive example. Real deployments can violate that assumption.
    This helper keeps the manuscript figures reproducible by falling back from the
    test split to the full window bank and, if necessary, to severity-ranked windows.
    """
    test_meta = meta_df.loc[test_mask].copy()

    normal_idx = select_representative_index(test_meta, label=0, quantile=0.50)
    alert_idx = select_representative_index(test_meta, label=1, quantile=0.75)

    normal_caption = "Normal window (test split)"
    alert_caption = "Pre-deterioration window (test split)"

    if normal_idx is None:
        normal_idx = select_representative_index(meta_df, label=0, quantile=0.50)
        if normal_idx is not None:
            normal_caption = "Normal window (global fallback)"

    if alert_idx is None:
        alert_idx = select_representative_index(meta_df, label=1, quantile=0.75)
        if alert_idx is not None:
            alert_caption = "Pre-deterioration window (global fallback)"

    if normal_idx is None:
        normal_idx = select_representative_index(meta_df, label=None, quantile=0.20)
        normal_caption = "Lower-risk window (score-ranked fallback)"

    if alert_idx is None:
        alert_idx = select_representative_index(meta_df, label=None, quantile=0.90)
        alert_caption = "Higher-risk window (score-ranked fallback)"

    if normal_idx is None or alert_idx is None:
        raise RuntimeError("Representative windows could not be selected from the available metadata.")

    if normal_idx == alert_idx and len(meta_df) > 1:
        ordered = meta_df["target_score"].sort_values()
        normal_idx = int(ordered.index[0])
        alert_idx = int(ordered.index[-1])
        normal_caption = "Lower-risk window (global fallback)"
        alert_caption = "Higher-risk window (global fallback)"

    return normal_idx, alert_idx, normal_caption, alert_caption



def annotate_empty_axis(ax: plt.Axes, title: str, message: str) -> None:
    """Create a non-crashing placeholder panel when data are unavailable."""
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=9, transform=ax.transAxes)



def plot_descriptive_results(
    df: pd.DataFrame,
    cfg: Config,
    results_root: Path,
    raw_bank: np.ndarray,
    topo_bank: np.ndarray,
    meta_df: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    """Create Section 5.1 descriptive tables and figures."""
    out = results_root / "5.1"
    ensure_dir(out)

    summary_rows = []
    for col in cfg.feature_vars:
        if col not in df.columns:
            continue
        summary_rows.append(
            {
                "Variable": col,
                "Mean": round(float(df[col].mean(skipna=True)), 3),
                "Std": round(float(df[col].std(skipna=True)), 3),
                "Min": round(float(df[col].min(skipna=True)), 3),
                "Max": round(float(df[col].max(skipna=True)), 3),
                "Missing_%": round(float(df[col].isna().mean() * 100), 2),
            }
        )
    table1 = pd.DataFrame(summary_rows)
    save_dataframe_table(
        table1,
        out / "table_1_dataset_summary.csv",
        out / "table_1_dataset_summary.png",
        "Table 1. Descriptive statistics of benchmark variables",
    )

    split_rows = []
    split_defs = {
        "Train": train_mask,
        "Validation": val_mask,
        "Test": test_mask,
    }
    for split_name, mask in split_defs.items():
        part = meta_df.loc[mask]
        split_rows.append(
            {
                "Split": split_name,
                "Windows": len(part),
                "EventRate": round(float(part["target_label"].mean()), 4),
                "MeanScore": round(float(part["target_score"].mean()), 4),
            }
        )
    table2 = pd.DataFrame(split_rows)
    save_dataframe_table(
        table2,
        out / "table_2_split_summary.csv",
        out / "table_2_split_summary.png",
        "Table 2. Event prevalence and mean severity by split",
    )

    complex_table = build_sensor_complex_report(df.ffill(), cfg.topo_vars)
    save_dataframe_table(
        complex_table,
        out / "table_3_sensor_complex_summary.csv",
        out / "table_3_sensor_complex_summary.png",
        "Table 3. Correlation-derived simplicial complex summary",
    )

    excerpt = df.iloc[max(0, len(df) // 4 - 80) : max(0, len(df) // 4 + 100)].copy()
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    series_info = [
        ("dissolved_oxygen_mgL", "Dissolved oxygen (mg/L)"),
        ("turbidity_ntu", "Turbidity (NTU)"),
        ("ammonia_mgL", "Ammonia (mg/L)"),
        ("e_coli_cfu100mL", "E. coli (CFU/100mL)"),
        ("rainfall_mm", "Rainfall (mm)"),
    ]
    for ax, (col, label) in zip(axes, series_info):
        if col not in excerpt.columns:
            continue
        ax.plot(excerpt["timestamp"], excerpt[col], linewidth=1.5)
        ax.set_ylabel(label, fontsize=9)
        if "event_label" in excerpt.columns:
            event_times = excerpt.loc[excerpt["event_label"] == 1, "timestamp"]
            for ts in event_times:
                ax.axvline(ts, alpha=0.03)
    axes[0].set_title("Figure 1. Representative multivariate benchmark segment")
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(out / "figure_1_multivariate_excerpt.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    normal_idx, alert_idx, normal_caption, alert_caption = choose_descriptive_windows(meta_df, test_mask)

    chosen_vars = [v for v in ["dissolved_oxygen_mgL", "turbidity_ntu", "e_coli_cfu100mL"] if v in cfg.topo_vars]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    feat_names = cfg.feature_vars + [f"{c}_mask" for c in cfg.feature_vars if f"{c}_mask" in df.columns]

    # Start with all panels disabled; individual panels are activated only when data exist.
    for ax in axes.ravel():
        ax.axis("off")

    representative_rows = [
        (normal_idx, normal_caption),
        (alert_idx, alert_caption),
    ]
    for row_i, (idx, label) in enumerate(representative_rows):
        raw_window = raw_bank[idx]
        axes[row_i, 0].axis("on")
        plotted_any_raw = False
        for col_name in ["dissolved_oxygen_mgL", "turbidity_ntu", "e_coli_cfu100mL"]:
            if col_name in feat_names:
                axes[row_i, 0].plot(raw_window[:, feat_names.index(col_name)], label=col_name)
                plotted_any_raw = True
        if plotted_any_raw:
            axes[row_i, 0].set_title(f"{label}: raw window")
            if row_i == 0:
                axes[row_i, 0].legend(fontsize=7)
        else:
            annotate_empty_axis(axes[row_i, 0], f"{label}: raw window", "Requested raw variables are not available.")

        for col_i, var in enumerate(chosen_vars, start=1):
            pos = cfg.topo_vars.index(var)
            channel = 2 * pos
            axes[row_i, col_i].axis("on")
            if channel < topo_bank.shape[1]:
                image = topo_bank[idx, channel]
                axes[row_i, col_i].imshow(image, aspect="auto")
                axes[row_i, col_i].set_title(f"{label}: {var} H0")
                axes[row_i, col_i].set_xticks([])
                axes[row_i, col_i].set_yticks([])
            else:
                annotate_empty_axis(axes[row_i, col_i], f"{label}: {var} H0", "Persistence channel unavailable.")

        for unused_col in range(1 + len(chosen_vars), axes.shape[1]):
            annotate_empty_axis(axes[row_i, unused_col], "Unused panel", "No additional persistence image was requested.")

    fig.suptitle("Figure 2. Normal versus pre-deterioration windows in raw and topological form", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "figure_2_normal_vs_alert_persistence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    test_indices = np.where(test_mask)[0]
    if len(test_indices) == 0:
        sample_idx = np.arange(min(len(meta_df), 80))
    elif len(test_indices) <= 80:
        sample_idx = test_indices
    else:
        sample_idx = np.linspace(test_indices[0], test_indices[-1], 80).astype(int)
        sample_idx = np.unique(sample_idx)

    if len(sample_idx) == 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        annotate_empty_axis(ax, "Figure 3. Mapper graph of persistence-image windows", "No windows were available to build the Mapper graph.")
        fig.tight_layout()
        fig.savefig(out / "figure_3_mapper_graph.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    X = topo_bank[sample_idx].reshape(len(sample_idx), -1)
    y = meta_df.iloc[sample_idx]["target_label"].values
    lens = PCA(n_components=2, random_state=cfg.seed).fit_transform(X)

    mapper = KeplerMapper(verbose=0)
    graph = mapper.map(lens, X, cover=km.Cover(n_cubes=7, perc_overlap=0.35), clusterer=DBSCAN(eps=1.3, min_samples=3))
    G = nx.Graph()
    node_event_rate = {}
    for node_name, members in graph["nodes"].items():
        G.add_node(node_name)
        node_event_rate[node_name] = float(np.mean(y[members])) if len(members) else 0.0
    for node_name, neighbors in graph["links"].items():
        for neighbor in neighbors:
            G.add_edge(node_name, neighbor)

    fig, ax = plt.subplots(figsize=(9, 7))
    if G.number_of_nodes() == 0:
        annotate_empty_axis(
            ax,
            "Figure 3. Mapper graph of persistence-image windows in the test split",
            "The Mapper configuration produced no nodes for the selected sample. Consider relaxing the cover or clustering parameters.",
        )
    else:
        pos = nx.spring_layout(G, seed=cfg.seed)
        node_sizes = [80 + 20 * len(graph["nodes"].get(node, [])) for node in G.nodes()]
        node_colors = [node_event_rate[node] for node in G.nodes()]
        nx.draw_networkx_edges(G, pos, alpha=0.35, ax=ax)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
        plt.colorbar(nodes, ax=ax, fraction=0.03, pad=0.02, label="Mean event label")
        ax.set_title("Figure 3. Mapper graph of persistence-image windows in the test split")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "figure_3_mapper_graph.png", dpi=220, bbox_inches="tight")
    plt.close(fig)




def safe_curve_arrays(y_true: np.ndarray, prob: np.ndarray, curve_type: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return plotting arrays for ROC or PR curves only when both classes exist."""
    y, p = sanitize_binary_targets_and_scores(y_true, prob)
    if len(y) == 0 or not has_two_classes(y):
        return None
    if curve_type == "roc":
        x, yy, _ = roc_curve(y, p)
        return x, yy
    if curve_type == "pr":
        precision, recall, _ = precision_recall_curve(y, p)
        return recall, precision
    raise ValueError(f"Unsupported curve_type: {curve_type}")




def finalize_curve_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, has_lines: bool, empty_message: str) -> None:
    """Finalize a metric-comparison axis even when no valid curves can be drawn."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if has_lines:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, empty_message, ha="center", va="center", wrap=True, fontsize=10, transform=ax.transAxes)




def plot_model_results(results: Dict[str, Dict], results_root: Path) -> None:
    """Create Section 5.2 predictive comparison tables and figures."""
    out = results_root / "5.2"
    ensure_dir(out)

    rows = []
    for model_name, result in results.items():
        cls = result["test_class"]
        reg = result["test_reg"]
        rows.append(
            {
                "Model": model_name,
                "AUROC": round(cls["AUROC"], 4),
                "AUPRC": round(cls["AUPRC"], 4),
                "F1": round(cls["F1"], 4),
                "Precision": round(cls["Precision"], 4),
                "Recall": round(cls["Recall"], 4),
                "BalancedAcc": round(cls["BalancedAcc"], 4),
                "MAE": round(reg["MAE"], 4),
                "RMSE": round(reg["RMSE"], 4),
                "R2": round(reg["R2"], 4),
                "Threshold": round(result["threshold"], 4),
            }
        )
    metrics_df = pd.DataFrame(rows)
    save_dataframe_table(
        metrics_df,
        out / "table_4_model_metrics.csv",
        out / "table_4_model_metrics.png",
        "Table 4. Test-set predictive performance of competing models",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    has_roc_lines = False
    for model_name, result in results.items():
        pred = result["predictions"]["test"]
        roc_arrays = safe_curve_arrays(pred["y_cls"], pred["prob"], curve_type="roc")
        if roc_arrays is None:
            continue
        fpr, tpr = roc_arrays
        ax.plot(fpr, tpr, label=f"{model_name} (AUROC={result['test_class']['AUROC']:.3f})")
        has_roc_lines = True
    if has_roc_lines:
        ax.plot([0, 1], [0, 1], linestyle="--")
    finalize_curve_axis(
        ax,
        title="Figure 4. ROC comparison of raw, topological, and fused models",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        has_lines=has_roc_lines,
        empty_message="ROC curves are undefined because the evaluated split does not contain both classes.",
    )
    fig.tight_layout()
    fig.savefig(out / "figure_4_roc_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    has_pr_lines = False
    for model_name, result in results.items():
        pred = result["predictions"]["test"]
        pr_arrays = safe_curve_arrays(pred["y_cls"], pred["prob"], curve_type="pr")
        if pr_arrays is None:
            continue
        recall, precision = pr_arrays
        ax.plot(recall, precision, label=f"{model_name} (AUPRC={result['test_class']['AUPRC']:.3f})")
        has_pr_lines = True
    finalize_curve_axis(
        ax,
        title="Figure 5. Precision-recall comparison of competing models",
        xlabel="Recall",
        ylabel="Precision",
        has_lines=has_pr_lines,
        empty_message="Precision-recall curves are undefined because the evaluated split does not contain both classes.",
    )
    fig.tight_layout()
    fig.savefig(out / "figure_5_pr_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, max(len(results), 1), figsize=(5.0 * max(len(results), 1), 4.8), sharex=False, sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (model_name, result) in zip(axes, results.items()):
        pred = result["predictions"]["test"]
        y_obs, y_hat = sanitize_regression_targets_and_predictions(pred["y_reg"], pred["reg_pred"])
        if len(y_obs) == 0:
            annotate_empty_axis(
                ax,
                f"{model_name} (R2=nan)",
                "No finite regression pairs were available for plotting.",
            )
            continue
        ax.scatter(y_obs, y_hat, alpha=0.60, s=18)
        min_v = min(float(np.min(y_obs)), float(np.min(y_hat)))
        max_v = max(float(np.max(y_obs)), float(np.max(y_hat)))
        ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
        ax.set_title(f"{model_name} (R2={result['test_reg']['R2']:.3f})")
        ax.set_xlabel("Observed score")
        ax.set_ylabel("Predicted score")

    fig.suptitle("Figure 6. Regression agreement for deterioration severity forecasts", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "figure_6_regression_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)




def evaluate_with_modified_inputs(
    model: nn.Module,
    loader: DataLoader,
    device: Any,
    raw_modifier=None,
    topo_modifier=None,
) -> Dict[str, np.ndarray]:
    """Evaluate a trained model after applying inference-time perturbations."""
    model.eval()
    reg_list, logit_list, y_reg_list, y_cls_list = [], [], [], []
    with torch.no_grad():
        for raw, topo, y_reg, y_cls in loader:
            if raw_modifier is not None:
                raw = raw_modifier(raw.clone())
            if topo_modifier is not None:
                topo = topo_modifier(topo.clone())
            raw = move_to_device(raw, device)
            topo = move_to_device(topo, device)
            reg_pred, cls_logit = model(raw, topo)
            reg_list.append(reg_pred.detach().cpu().numpy().ravel())
            logit_list.append(cls_logit.detach().cpu().numpy().ravel())
            y_reg_list.append(y_reg.numpy().ravel())
            y_cls_list.append(y_cls.numpy().ravel())
    if len(reg_list) == 0:
        empty = np.array([], dtype=np.float32)
        return {"reg_pred": empty, "prob": empty, "y_reg": empty, "y_cls": empty}
    reg_pred = np.nan_to_num(np.concatenate(reg_list), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    logits = np.nan_to_num(np.concatenate(logit_list), nan=0.0, posinf=50.0, neginf=-50.0).astype(np.float32)
    y_reg = np.nan_to_num(np.concatenate(y_reg_list), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    y_cls = np.clip(np.nan_to_num(np.concatenate(y_cls_list), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    return {"reg_pred": reg_pred, "prob": prob, "y_reg": y_reg, "y_cls": y_cls}




def plot_robustness_and_ablation(
    fusion_model: nn.Module,
    fusion_result: Dict,
    test_loader: DataLoader,
    cfg: Config,
    results_root: Path,
    raw_feature_names: List[str],
    topo_var_names: List[str],
    device: Any,
) -> None:
    """Create Section 5.3 robustness and ablation outputs."""
    out = results_root / "5.3"
    ensure_dir(out)

    baseline_threshold = fusion_result["threshold"]
    baseline_pred = fusion_result["predictions"]["test"]

    ablation_rows = []

    def zero_raw(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def zero_topo(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def zero_h0(x: torch.Tensor) -> torch.Tensor:
        x[:, ::2, :, :] = 0.0
        return x

    def zero_h1(x: torch.Tensor) -> torch.Tensor:
        x[:, 1::2, :, :] = 0.0
        return x

    perturbations = {
        "Baseline": (None, None),
        "Raw branch removed": (zero_raw, None),
        "Topological branch removed": (None, zero_topo),
        "H0 channels removed": (None, zero_h0),
        "H1 channels removed": (None, zero_h1),
    }

    for name, (raw_mod, topo_mod) in perturbations.items():
        pred = (
            baseline_pred
            if name == "Baseline"
            else evaluate_with_modified_inputs(fusion_model, test_loader, device, raw_modifier=raw_mod, topo_modifier=topo_mod)
        )
        cls = classification_metrics(pred["y_cls"], pred["prob"], baseline_threshold)
        reg = regression_metrics(pred["y_reg"], pred["reg_pred"])
        ablation_rows.append(
            {
                "Setting": name,
                "AUROC": round(cls["AUROC"], 4),
                "AUPRC": round(cls["AUPRC"], 4),
                "F1": round(cls["F1"], 4),
                "Recall": round(cls["Recall"], 4),
                "RMSE": round(reg["RMSE"], 4),
            }
        )

    ablation_df = pd.DataFrame(ablation_rows)
    save_dataframe_table(
        ablation_df,
        out / "table_5_ablation_analysis.csv",
        out / "table_5_ablation_analysis.png",
        "Table 5. Fusion-model ablation analysis on the test split",
    )

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    noise_records = []
    rng = np.random.default_rng(cfg.seed)

    for noise in noise_levels:
        def noisy_raw(x: torch.Tensor, sigma: float = noise) -> torch.Tensor:
            if sigma == 0.0:
                return x
            arr = x.numpy()
            arr = arr + rng.normal(0.0, sigma, size=arr.shape)
            return torch.tensor(arr, dtype=torch.float32)

        def noisy_topo(x: torch.Tensor, sigma: float = noise) -> torch.Tensor:
            if sigma == 0.0:
                return x
            arr = x.numpy()
            arr = arr + rng.normal(0.0, sigma, size=arr.shape)
            return torch.tensor(arr, dtype=torch.float32)

        pred = evaluate_with_modified_inputs(fusion_model, test_loader, device, raw_modifier=noisy_raw, topo_modifier=noisy_topo)
        cls = classification_metrics(pred["y_cls"], pred["prob"], baseline_threshold)
        noise_records.append({"NoiseLevel": noise, "AUROC": cls["AUROC"], "F1": cls["F1"]})

    noise_df = pd.DataFrame(noise_records)
    noise_df.to_csv(out / "noise_robustness.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(noise_df["NoiseLevel"], noise_df["AUROC"], marker="o", label="AUROC")
    ax1.set_xlabel("Gaussian corruption level")
    ax1.set_ylabel("AUROC")
    ax1.set_title("Figure 7. Noise robustness of the fusion model")
    ax2 = ax1.twinx()
    ax2.plot(noise_df["NoiseLevel"], noise_df["F1"], marker="s", linestyle="--", label="F1")
    ax2.set_ylabel("F1")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out / "figure_7_noise_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Variable importance by leave-one-variable-out perturbation.
    importance_rows = []
    n_feature_channels = len(raw_feature_names)
    n_topo_channels = len(topo_var_names) * 2
    baseline_auroc = classification_metrics(baseline_pred["y_cls"], baseline_pred["prob"], baseline_threshold)["AUROC"]

    for var in topo_var_names:
        raw_indices = [i for i, name in enumerate(raw_feature_names) if name == var or name == f"{var}_mask"]
        topo_index = topo_var_names.index(var)
        topo_channels = [2 * topo_index, 2 * topo_index + 1]

        def var_raw_modifier(x: torch.Tensor, idxs=raw_indices) -> torch.Tensor:
            for idx in idxs:
                if idx < x.shape[-1]:
                    x[:, :, idx] = 0.0
            return x

        def var_topo_modifier(x: torch.Tensor, chs=topo_channels) -> torch.Tensor:
            for ch in chs:
                if ch < x.shape[1]:
                    x[:, ch, :, :] = 0.0
            return x

        pred = evaluate_with_modified_inputs(fusion_model, test_loader, device, raw_modifier=var_raw_modifier, topo_modifier=var_topo_modifier)
        cls = classification_metrics(pred["y_cls"], pred["prob"], baseline_threshold)
        importance_rows.append({"Variable": var, "Delta_AUROC": baseline_auroc - cls["AUROC"]})

    importance_df = pd.DataFrame(importance_rows).sort_values("Delta_AUROC", ascending=False)
    importance_df.to_csv(out / "variable_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(importance_df["Variable"], importance_df["Delta_AUROC"])
    ax.set_ylabel("Decrease in AUROC after variable removal")
    ax.set_xlabel("Variable")
    ax.set_title("Figure 8. Variable contribution to fusion-model discrimination")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "figure_8_variable_importance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------



def main() -> None:
    parser = argparse.ArgumentParser(description="Topological deep learning for water-quality deterioration forecasting")
    parser.add_argument("--data_csv", type=str, default=None, help="Path to an input CSV file with a timestamp column")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory where subsection outputs will be stored")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "directml"],
        help="Computation backend. Auto prefers CUDA, then DirectML, then CPU.",
    )
    parser.add_argument("--config_json", type=str, default=None, help="Optional JSON file overriding configuration values")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs

    if args.config_json is not None:
        with open(args.config_json, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        for key, value in user_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    set_seed(cfg.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "5.1")
    ensure_dir(output_dir / "5.2")
    ensure_dir(output_dir / "5.3")

    device, device_info = select_compute_device(args.device)
    print(f"[INFO] Selected backend: {device_info['selected_backend']} ({device_info['device_repr']})")
    if device_info.get("notes"):
        for note in device_info["notes"]:
            print(f"[INFO] {note}")

    save_json(device_info, output_dir / "device_info.json")
    save_json(asdict(cfg), output_dir / "effective_config.json")

    print("[INFO] Loading or generating dataset...")
    df = load_input_dataset(cfg, args.data_csv)

    print("[INFO] Computing deterioration targets...")
    df = compute_deterioration_targets(df, cfg)

    print("[INFO] Preprocessing time series...")
    processed_df, medians, iqrs, active_feature_vars = preprocess_time_series(df, cfg)
    save_json(medians, output_dir / "feature_medians.json")
    save_json(iqrs, output_dir / "feature_iqrs.json")

    print("[INFO] Building raw and topological windows...")
    raw_bank, topo_bank, meta_df, active_mask_vars = build_window_bank(processed_df, cfg, active_feature_vars)
    np.save(output_dir / "raw_bank.npy", raw_bank)
    np.save(output_dir / "topo_bank.npy", topo_bank)
    meta_df.to_csv(output_dir / "window_metadata.csv", index=False)

    n_train = int(len(processed_df) * cfg.train_frac)
    n_val = int(len(processed_df) * cfg.val_frac)
    train_mask = meta_df["target_idx"].values < n_train
    val_mask = (meta_df["target_idx"].values >= n_train) & (meta_df["target_idx"].values < n_train + n_val)
    test_mask = meta_df["target_idx"].values >= n_train + n_val

    split_summary = {
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(val_mask.sum()),
        "test_windows": int(test_mask.sum()),
    }
    save_json(split_summary, output_dir / "split_summary.json")
    print(
        "[INFO] Window split sizes -> "
        f"train: {split_summary['train_windows']}, "
        f"validation: {split_summary['validation_windows']}, "
        f"test: {split_summary['test_windows']}"
    )

    y_reg = np.nan_to_num(meta_df["target_score"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_cls = np.clip(np.nan_to_num(meta_df["target_label"].values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    train_ds = WindowDataset(raw_bank[train_mask], topo_bank[train_mask], y_reg[train_mask], y_cls[train_mask])
    val_ds = WindowDataset(raw_bank[val_mask], topo_bank[val_mask], y_reg[val_mask], y_cls[val_mask])
    test_ds = WindowDataset(raw_bank[test_mask], topo_bank[test_mask], y_reg[test_mask], y_cls[test_mask])

    pin_memory = bool(device_info["selected_backend"] == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory)

    print("[INFO] Exporting Section 5.1 descriptive artifacts...")
    plot_descriptive_results(df, cfg, output_dir, raw_bank, topo_bank, meta_df, train_mask, val_mask, test_mask)

    raw_dim = raw_bank.shape[-1]
    topo_channels = topo_bank.shape[1]

    print("[INFO] Training raw-only model...")
    raw_model = RawOnlyNet(raw_dim)
    raw_model, raw_result = train_model(raw_model, train_loader, val_loader, test_loader, cfg, device)

    print("[INFO] Training topology-only model...")
    topo_model = TopologyOnlyNet(topo_channels)
    topo_model, topo_result = train_model(topo_model, train_loader, val_loader, test_loader, cfg, device)

    print("[INFO] Training fusion model...")
    fusion_model = FusionNet(raw_dim, topo_channels)
    fusion_model, fusion_result = train_model(fusion_model, train_loader, val_loader, test_loader, cfg, device)

    all_results = {
        "RawOnly": raw_result,
        "TopoOnly": topo_result,
        "Fusion": fusion_result,
    }

    compact = {}
    for name, result in all_results.items():
        compact[name] = {
            "threshold": result["threshold"],
            "train_class": result["train_class"],
            "val_class": result["val_class"],
            "test_class": result["test_class"],
            "train_reg": result["train_reg"],
            "val_reg": result["val_reg"],
            "test_reg": result["test_reg"],
        }
    save_json(compact, output_dir / "model_metric_summary.json")

    print("[INFO] Exporting Section 5.2 model-comparison artifacts...")
    plot_model_results(all_results, output_dir)

    print("[INFO] Exporting Section 5.3 robustness and ablation artifacts...")
    raw_feature_names = active_feature_vars + active_mask_vars
    active_topo_vars = [col for col in cfg.topo_vars if col in processed_df.columns]
    plot_robustness_and_ablation(
        fusion_model,
        fusion_result,
        test_loader,
        cfg,
        output_dir,
        raw_feature_names=raw_feature_names,
        topo_var_names=active_topo_vars,
        device=device,
    )

    print("[INFO] Saving trained model checkpoints...")
    torch.save(raw_model.state_dict(), output_dir / "raw_only_model.pt")
    torch.save(topo_model.state_dict(), output_dir / "topology_only_model.pt")
    torch.save(fusion_model.state_dict(), output_dir / "fusion_model.pt")

    print("[DONE] Pipeline completed successfully.")
    print(f"Outputs are available under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
