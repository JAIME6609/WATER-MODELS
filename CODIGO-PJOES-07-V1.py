
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
urban_water_tdl_pipeline.py

Explainable Topological Deep Learning for Pollution Risk Mapping
in Multi-Source Urban Water Systems

This script operationalizes the conceptual and computational framework
proposed in Sections 1–4 of the article. It is intentionally designed
to support the later writing of Sections 5.1, 5.2, and 5.3 by producing:

    outputs/5.1 -> predictive performance tables and figures
    outputs/5.2 -> topological signature interpretation artifacts
    outputs/5.3 -> intervention prioritization and scenario-analysis artifacts

The code is complete and functional, but it requires specialized topology
libraries that are usually not preinstalled in minimal Python environments:
Ripser, Persim, KeplerMapper (kmapper), and TopoNetX.

No part of this code uses GUDHI, in accordance with the requested constraint.

The pipeline supports two modes:
    1. Real data mode: the user provides nodes and edges CSV files.
    2. Demo mode: if the configured files do not exist and demo generation
       is enabled, a synthetic multi-source urban water system is generated.

The full workflow covers:
    - ingestion and preprocessing of multi-source water data,
    - multiplex graph construction,
    - local persistent homology and Hodge-inspired topological descriptors,
    - topology-aware deep learning with PyTorch,
    - feature attribution and topological attribution,
    - extraction of risk pathways,
    - intervention-priority ranking,
    - scenario-based comparative analysis.

The code is heavily commented so that every important design choice remains
traceable and easy to adapt to a real empirical case study.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Optional topology libraries.
# The script compiles and can be inspected even when these packages are absent.
# Functions that require them fail with explicit and actionable error messages.
# --------------------------------------------------------------------------------------

try:
    from ripser import ripser
except Exception:
    ripser = None

try:
    from persim import PersistenceImager
except Exception:
    PersistenceImager = None

try:
    import kmapper as km
except Exception:
    km = None

try:
    import toponetx as tnx  # optional; manual fallback is provided
except Exception:
    tnx = None

try:
    import yaml
except Exception:
    yaml = None


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Container for all node-level analytical objects after preprocessing."""
    nodes_df: pd.DataFrame
    feature_matrix: np.ndarray
    feature_names: List[str]
    labels: np.ndarray
    node_ids: List[str]
    source_type_ids: np.ndarray
    source_type_names: List[str]
    coords: np.ndarray
    exposure: np.ndarray
    mitigation: np.ndarray
    groups: np.ndarray
    metadata_columns: List[str]


@dataclass
class GraphBundle:
    """Container for graph-level objects used by topology, learning, and explanation."""
    relation_graphs: Dict[str, nx.DiGraph]
    aggregate_graph: nx.DiGraph
    adjacency_by_relation: Dict[str, np.ndarray]
    aggregate_adjacency: np.ndarray
    graph_stats: pd.DataFrame
    relation_names: List[str]
    edge_df: pd.DataFrame


@dataclass
class TopologyBundle:
    """Topological descriptors and artifacts computed from node neighborhoods."""
    topo_matrix: np.ndarray
    topo_feature_names: List[str]
    diagrams: Dict[str, Dict[int, np.ndarray]]
    betti_curves: Dict[str, Dict[int, np.ndarray]]
    persistence_images: Dict[str, Dict[int, np.ndarray]]
    hodge_stats: Dict[str, np.ndarray]
    neighborhoods: Dict[str, List[str]]
    representative_nodes: List[str]


@dataclass
class TrainingArtifacts:
    """Trained model and diagnostic history."""
    model: nn.Module
    history: pd.DataFrame
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    test_probabilities: np.ndarray
    test_predictions: np.ndarray
    full_probabilities: np.ndarray
    latent_embeddings: np.ndarray
    threshold: float


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def setup_logger(output_dir: Path) -> None:
    """Configure file and console logging in a reproducible way."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def set_seed(seed: int) -> None:
    """Fix all major random seeds to improve reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_topology_dependencies() -> None:
    """
    Stop execution with a clear message when the indispensable topology package
    is not available. Ripser is required because persistent homology is central
    to the requested framework.

    Persim, KeplerMapper, and TopoNetX are used when available, but the code
    contains documented fallbacks for some of their roles so the pipeline
    remains easier to adapt across environments.
    """
    if ripser is None:
        raise ImportError(
            "ripser is required but not installed. Install the packages listed "
            "in requirements.txt and rerun the pipeline."
        )


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML or JSON configuration. YAML is preferred because it is more readable
    for scientific pipelines that contain nested experiment settings.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in [".yaml", ".yml"]:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configuration files.")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError("Configuration file must be YAML/YML or JSON.")


def safe_mkdir(path: Path) -> None:
    """Create a directory only when needed."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    """Write JSON with indentation for human readability."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def robust_zscore(series: np.ndarray) -> np.ndarray:
    """
    Robust normalization based on median and IQR. This is useful in urban water data,
    where outliers may reflect either true contamination events or monitoring artifacts.
    """
    series = np.asarray(series, dtype=float)
    med = np.nanmedian(series)
    q75, q25 = np.nanpercentile(series, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        iqr = np.nanstd(series) + 1e-8
    return (series - med) / (iqr + 1e-8)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    """Simple min-max scaling used only for visualization and ranking support."""
    x = np.asarray(x, dtype=float)
    if np.allclose(np.nanmax(x), np.nanmin(x)):
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-8)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Row-normalize an adjacency matrix for message passing. This creates
    a weighted neighborhood averaging operator.
    """
    matrix = matrix.copy().astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums


def write_text(path: Path, content: str) -> None:
    """Small helper used to export textual summaries and audit trails."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def select_prefixed_columns(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    """Return all columns whose names start with any of the supplied prefixes."""
    selected = []
    for col in df.columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                selected.append(col)
                break
    return selected


def build_feature_group_map(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract prefix conventions from configuration. The article framework divides
    the node representation into chemistry, hydraulics, hydrogeology/geospatial,
    operations, and temporal context.
    """
    data_cfg = cfg["data"]
    return {
        "chem": data_cfg.get("chem_prefix", "chem_"),
        "hyd": data_cfg.get("hyd_prefix", "hyd_"),
        "geo": data_cfg.get("geo_prefix", "geo_"),
        "ops": data_cfg.get("ops_prefix", "ops_"),
        "tmp": data_cfg.get("tmp_prefix", "tmp_"),
    }


def default_device() -> torch.device:
    """Use GPU when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# Synthetic demo-data generation
# --------------------------------------------------------------------------------------

def generate_demo_dataset(nodes_path: Path, edges_path: Path, seed: int = 42) -> None:
    """
    Generate a synthetic multi-source urban water system. The goal is not to emulate
    a specific city, but to provide a realistic enough structure so the user can test
    the full pipeline before replacing the demo data with a real case study.

    The synthetic network includes four source domains:
        - surface water,
        - groundwater,
        - reused water,
        - cross-source infrastructure/interaction nodes.

    The synthetic labels are derived from a latent contamination mechanism influenced by:
        - chemical pressure,
        - operational weakness,
        - hydrogeological susceptibility,
        - source bridging,
        - temporal exceedance memory.
    """
    rng = np.random.default_rng(seed)

    # We generate spatial clusters so that later spatial blocking has something meaningful to use.
    source_types = ["surface", "groundwater", "reuse", "cross"]
    n_by_source = {"surface": 55, "groundwater": 55, "reuse": 40, "cross": 30}
    centers = {
        "surface": np.array([20.0, 80.0]),
        "groundwater": np.array([75.0, 25.0]),
        "reuse": np.array([80.0, 80.0]),
        "cross": np.array([50.0, 50.0]),
    }

    rows = []
    node_counter = 0
    for source in source_types:
        for _ in range(n_by_source[source]):
            node_counter += 1
            node_id = f"N{node_counter:03d}"
            center = centers[source]
            x, y = rng.normal(center[0], 10.0), rng.normal(center[1], 10.0)
            district = int(np.clip((x // 20) + 5 * (y // 20), 0, 24))

            # Chemistry block
            chem_nitrate = abs(rng.normal(8.0 if source != "reuse" else 10.0, 3.0))
            chem_ammonium = abs(rng.normal(0.35 if source != "reuse" else 0.55, 0.18))
            chem_conductivity = abs(rng.normal(850 if source == "groundwater" else 650, 120))
            chem_ecoli = abs(rng.normal(0.8 if source == "reuse" else 0.2, 0.6))
            chem_turbidity = abs(rng.normal(3.0 if source == "surface" else 1.5, 1.2))

            # Hydraulic block
            hyd_flow = abs(rng.normal(55 if source == "surface" else 35, 15))
            hyd_pressure = abs(rng.normal(42 if source in ["reuse", "cross"] else 28, 8))
            hyd_residence_time = abs(rng.normal(9 if source == "groundwater" else 5, 2.5))
            hyd_pumping_intensity = abs(rng.normal(7 if source == "groundwater" else 4, 2))

            # Hydrogeological / geospatial block
            geo_recharge = abs(rng.normal(0.55 if source == "groundwater" else 0.35, 0.15))
            geo_depth_gw = abs(rng.normal(28 if source == "groundwater" else 14, 8))
            geo_permeability = abs(rng.normal(0.65 if source in ["groundwater", "cross"] else 0.35, 0.15))
            geo_landuse_pressure = abs(rng.normal(0.7 if source in ["surface", "reuse"] else 0.45, 0.18))
            geo_elevation = abs(rng.normal(1200, 40))

            # Operational block
            ops_asset_age = abs(rng.normal(22 if source in ["reuse", "cross"] else 18, 7))
            ops_treatment_score = abs(rng.normal(0.55 if source == "reuse" else 0.72, 0.15))
            ops_leakage_proxy = abs(rng.normal(0.18 if source != "surface" else 0.12, 0.08))
            ops_maintenance_gap = abs(rng.normal(0.28 if source in ["cross", "reuse"] else 0.18, 0.10))

            # Temporal block
            tmp_exceedance_memory = abs(rng.normal(0.32 if source in ["reuse", "cross"] else 0.20, 0.12))
            tmp_exceedance_frequency = abs(rng.normal(0.24 if source in ["reuse", "surface"] else 0.16, 0.10))
            tmp_rain_event_marker = rng.integers(0, 2)
            tmp_seasonality_index = abs(rng.normal(0.50, 0.20))

            # Exposure and mitigation for intervention prioritization
            exposure_score = abs(rng.normal(0.70 if district in [6, 7, 11, 12] else 0.45, 0.20))
            mitigation_score = abs(rng.normal(0.45 if source in ["cross", "reuse"] else 0.60, 0.18))

            rows.append(
                {
                    "node_id": node_id,
                    "source_type": source,
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "district": f"D{district:02d}",
                    "chem_nitrate": round(float(chem_nitrate), 4),
                    "chem_ammonium": round(float(chem_ammonium), 4),
                    "chem_conductivity": round(float(chem_conductivity), 4),
                    "chem_ecoli": round(float(chem_ecoli), 4),
                    "chem_turbidity": round(float(chem_turbidity), 4),
                    "hyd_flow": round(float(hyd_flow), 4),
                    "hyd_pressure": round(float(hyd_pressure), 4),
                    "hyd_residence_time": round(float(hyd_residence_time), 4),
                    "hyd_pumping_intensity": round(float(hyd_pumping_intensity), 4),
                    "geo_recharge": round(float(geo_recharge), 4),
                    "geo_depth_gw": round(float(geo_depth_gw), 4),
                    "geo_permeability": round(float(geo_permeability), 4),
                    "geo_landuse_pressure": round(float(geo_landuse_pressure), 4),
                    "geo_elevation": round(float(geo_elevation), 4),
                    "ops_asset_age": round(float(ops_asset_age), 4),
                    "ops_treatment_score": round(float(ops_treatment_score), 4),
                    "ops_leakage_proxy": round(float(ops_leakage_proxy), 4),
                    "ops_maintenance_gap": round(float(ops_maintenance_gap), 4),
                    "tmp_exceedance_memory": round(float(tmp_exceedance_memory), 4),
                    "tmp_exceedance_frequency": round(float(tmp_exceedance_frequency), 4),
                    "tmp_rain_event_marker": int(tmp_rain_event_marker),
                    "tmp_seasonality_index": round(float(tmp_seasonality_index), 4),
                    "exposure_score": round(float(np.clip(exposure_score, 0.01, 1.25)), 4),
                    "mitigation_score": round(float(np.clip(mitigation_score, 0.01, 1.25)), 4),
                }
            )

    nodes_df = pd.DataFrame(rows)

    # Risk label generation:
    # A latent score combines multiple blocks in a way consistent with the article's logic.
    bridge_bonus = nodes_df["source_type"].isin(["cross", "reuse"]).astype(float) * 0.35
    latent_risk = (
        0.24 * robust_zscore(nodes_df["chem_nitrate"].values)
        + 0.16 * robust_zscore(nodes_df["chem_ecoli"].values)
        + 0.10 * robust_zscore(nodes_df["chem_ammonium"].values)
        + 0.14 * robust_zscore(nodes_df["ops_leakage_proxy"].values)
        + 0.08 * robust_zscore(nodes_df["ops_maintenance_gap"].values)
        + 0.11 * robust_zscore(nodes_df["geo_landuse_pressure"].values)
        + 0.08 * robust_zscore(nodes_df["geo_permeability"].values)
        + 0.12 * robust_zscore(nodes_df["tmp_exceedance_memory"].values)
        + 0.07 * robust_zscore(nodes_df["tmp_exceedance_frequency"].values)
        + bridge_bonus
        + np.random.default_rng(seed).normal(0, 0.12, size=len(nodes_df))
    )
    threshold = np.quantile(latent_risk, 0.70)
    nodes_df["risk_label"] = (latent_risk >= threshold).astype(int)
    nodes_df["latent_risk_demo_only"] = latent_risk

    # Build layered edges with different source semantics.
    edges = []
    # Use k-nearest neighbors within each source type.
    for source_type in source_types:
        sub = nodes_df[nodes_df["source_type"] == source_type].copy()
        coords = sub[["x", "y"]].values
        node_list = sub["node_id"].tolist()
        nbrs = NearestNeighbors(n_neighbors=min(6, len(sub))).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        relation_type = "surface" if source_type == "surface" else (
            "groundwater" if source_type == "groundwater" else (
                "reuse" if source_type == "reuse" else "cross"
            )
        )
        for i, src_id in enumerate(node_list):
            for d, j in zip(distances[i, 1:], indices[i, 1:]):
                tgt_id = node_list[j]
                edges.append(
                    {
                        "source": src_id,
                        "target": tgt_id,
                        "relation_type": relation_type,
                        "geographic_distance": float(d),
                        "hydro_distance": float(d * rng.uniform(0.6, 1.5)),
                        "chemical_distance": float(abs(sub.iloc[i]["chem_nitrate"] - sub.iloc[j]["chem_nitrate"])),
                        "reuse_distance": float(rng.uniform(0.1, 0.6) if relation_type == "reuse" else rng.uniform(0.5, 1.5)),
                    }
                )

    # Add cross-source connectors.
    cross_nodes = nodes_df[nodes_df["source_type"] == "cross"]["node_id"].tolist()
    non_cross = nodes_df[nodes_df["source_type"] != "cross"].copy()
    cross_df = nodes_df[nodes_df["source_type"] == "cross"].copy()
    for cross_id in cross_nodes:
        cross_row = cross_df[cross_df["node_id"] == cross_id].iloc[0]
        for source_type in ["surface", "groundwater", "reuse"]:
            sub = nodes_df[nodes_df["source_type"] == source_type].copy()
            coords = sub[["x", "y"]].values
            nbrs = NearestNeighbors(n_neighbors=min(3, len(sub))).fit(coords)
            dists, idxs = nbrs.kneighbors(np.array([[cross_row["x"], cross_row["y"]]]))
            for d, j in zip(dists[0], idxs[0]):
                tgt_id = sub.iloc[j]["node_id"]
                edges.append(
                    {
                        "source": cross_id,
                        "target": tgt_id,
                        "relation_type": "cross",
                        "geographic_distance": float(d),
                        "hydro_distance": float(d * rng.uniform(0.5, 1.7)),
                        "chemical_distance": float(abs(cross_row["chem_nitrate"] - sub.iloc[j]["chem_nitrate"])),
                        "reuse_distance": float(rng.uniform(0.15, 0.8)),
                    }
                )
                # Reverse relation for bidirectional coupling.
                edges.append(
                    {
                        "source": tgt_id,
                        "target": cross_id,
                        "relation_type": "cross",
                        "geographic_distance": float(d),
                        "hydro_distance": float(d * rng.uniform(0.5, 1.7)),
                        "chemical_distance": float(abs(cross_row["chem_nitrate"] - sub.iloc[j]["chem_nitrate"])),
                        "reuse_distance": float(rng.uniform(0.15, 0.8)),
                    }
                )

    edges_df = pd.DataFrame(edges).drop_duplicates(subset=["source", "target", "relation_type"])
    safe_mkdir(nodes_path.parent)
    safe_mkdir(edges_path.parent)
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    logging.info("Demo dataset generated at %s and %s", nodes_path, edges_path)


# --------------------------------------------------------------------------------------
# Data loading and preprocessing
# --------------------------------------------------------------------------------------

def construct_label_from_thresholds(df: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    """
    Build a binary critical-risk target from pollutant concentrations and exceedance frequency.

    This implements the methodological logic described in the article:
    the label is positive when at least one configured pollutant exceeds its threshold
    and exceedance recurrence is sufficiently frequent.

    If the configuration is incomplete, the function falls back to a weaker but still
    reasonable rule based only on available pollutant columns.
    """
    data_cfg = cfg["data"]
    thresholds = data_cfg.get("pollutant_thresholds", {})
    exceed_col = data_cfg.get("exceedance_frequency_col", "tmp_exceedance_frequency")
    min_freq = float(data_cfg.get("min_exceedance_frequency", 0.2))

    if not thresholds:
        # Fallback: use chemistry columns and a high-quantile event criterion.
        chem_cols = select_prefixed_columns(df, [data_cfg.get("chem_prefix", "chem_")])
        if not chem_cols:
            raise ValueError(
                "No pollutant thresholds and no chemistry columns were found. "
                "The target label cannot be constructed automatically."
            )
        scores = np.zeros(len(df), dtype=float)
        for col in chem_cols:
            scores += normalize_minmax(df[col].values)
        freq = df[exceed_col].values if exceed_col in df.columns else np.zeros(len(df))
        label = (scores >= np.quantile(scores, 0.75)) & (freq >= np.quantile(freq, 0.60))
        return label.astype(int)

    exceed_mask = np.zeros(len(df), dtype=bool)
    for col, thr in thresholds.items():
        if col in df.columns:
            exceed_mask = exceed_mask | (df[col].astype(float).values > float(thr))

    if exceed_col in df.columns:
        freq_mask = df[exceed_col].astype(float).values >= min_freq
    else:
        freq_mask = np.ones(len(df), dtype=bool)

    return (exceed_mask & freq_mask).astype(int)


def load_and_preprocess_data(cfg: Dict[str, Any]) -> DataBundle:
    """
    Load node data, create labels when needed, standardize continuous features,
    encode source types, and return a DataBundle ready for graph construction
    and deep learning.
    """
    data_cfg = cfg["data"]
    nodes_path = Path(data_cfg["nodes_csv"])

    if not nodes_path.exists():
        if cfg["project"].get("generate_demo_data_if_missing", True):
            generate_demo_dataset(nodes_path, Path(data_cfg["edges_csv"]), cfg["project"]["seed"])
        else:
            raise FileNotFoundError(f"Nodes CSV not found: {nodes_path}")

    df = pd.read_csv(nodes_path)
    df = df.copy()

    # Standard column names expected by the rest of the pipeline.
    node_id_col = data_cfg.get("node_id_col", "node_id")
    source_type_col = data_cfg.get("source_type_col", "source_type")
    x_col = data_cfg.get("x_coord_col", "x")
    y_col = data_cfg.get("y_coord_col", "y")
    label_col = data_cfg.get("label_col", "risk_label")
    exposure_col = data_cfg.get("exposure_col", "exposure_score")
    mitigation_col = data_cfg.get("mitigation_col", "mitigation_score")
    group_col = data_cfg.get("group_col", "district")

    for required in [node_id_col, source_type_col, x_col, y_col]:
        if required not in df.columns:
            raise ValueError(f"Required column '{required}' not found in nodes CSV.")

    if label_col not in df.columns:
        df[label_col] = construct_label_from_thresholds(df, cfg)

    if exposure_col not in df.columns:
        df[exposure_col] = 0.5
    if mitigation_col not in df.columns:
        df[mitigation_col] = 0.5
    if group_col not in df.columns:
        # If administrative or district group is absent, use a temporary placeholder.
        df[group_col] = "G0"

    # Internally, the pipeline standardizes core metadata names so that all
    # downstream functions can remain concise and predictable even when the
    # external input files use other column names.
    rename_standard = {
        node_id_col: "node_id",
        source_type_col: "source_type",
        x_col: "x",
        y_col: "y",
        group_col: "district",
        label_col: "risk_label",
        exposure_col: "exposure_score",
        mitigation_col: "mitigation_score",
    }
    df = df.rename(columns=rename_standard)

    # Identify structured feature groups using the article's notation.
    prefixes = build_feature_group_map(cfg)
    continuous_cols = select_prefixed_columns(df, list(prefixes.values()))

    if not continuous_cols:
        raise ValueError(
            "No analytical feature columns were found. "
            "Expected columns with prefixes such as chem_, hyd_, geo_, ops_, tmp_."
        )

    # Robust imputation and scaling keep the pipeline stable under realistic missingness.
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    feature_matrix = scaler.fit_transform(imputer.fit_transform(df[continuous_cols]))
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

    # Encode source types because the model uses a source-context embedding.
    source_type_names = sorted(df["source_type"].astype(str).unique().tolist())
    source_to_id = {name: i for i, name in enumerate(source_type_names)}
    source_type_ids = df["source_type"].astype(str).map(source_to_id).values.astype(int)

    coords = df[["x", "y"]].values.astype(np.float32)
    labels = df["risk_label"].astype(int).values
    exposure = df["exposure_score"].astype(float).values
    mitigation = df["mitigation_score"].astype(float).values
    groups = df["district"].astype(str).values
    node_ids = df["node_id"].astype(str).tolist()

    # Metadata columns are kept for later tables and plots.
    metadata_columns = ["node_id", "source_type", "x", "y", "district"]

    logging.info("Loaded %d nodes and %d analytical features.", len(df), feature_matrix.shape[1])

    return DataBundle(
        nodes_df=df,
        feature_matrix=feature_matrix,
        feature_names=continuous_cols,
        labels=labels,
        node_ids=node_ids,
        source_type_ids=source_type_ids,
        source_type_names=source_type_names,
        coords=coords,
        exposure=exposure,
        mitigation=mitigation,
        groups=groups,
        metadata_columns=metadata_columns,
    )


# --------------------------------------------------------------------------------------
# Multiplex graph construction
# --------------------------------------------------------------------------------------

def compute_edge_weight(
    src_row: pd.Series,
    tgt_row: pd.Series,
    edge_row: Optional[pd.Series],
    cfg: Dict[str, Any],
) -> float:
    """
    Compute the affinity weight of an edge from a composite dissimilarity.
    This mirrors the methodological formulation in Section 3.1.

    The code uses:
        - geographic dissimilarity,
        - hydro / hydrogeological dissimilarity,
        - chemical dissimilarity,
        - reuse-related dissimilarity.

    If the edge file already contains these distances, the code uses them.
    Otherwise, it derives approximate dissimilarities directly from node features.
    """
    weights = cfg["graph"].get("similarity_weights", {
        "geographic": 0.25, "hydro": 0.25, "chemical": 0.25, "reuse": 0.25
    })

    geo = hyd = chem = reuse = None

    if edge_row is not None:
        if "geographic_distance" in edge_row.index:
            geo = float(edge_row["geographic_distance"])
        if "hydro_distance" in edge_row.index:
            hyd = float(edge_row["hydro_distance"])
        if "chemical_distance" in edge_row.index:
            chem = float(edge_row["chemical_distance"])
        if "reuse_distance" in edge_row.index:
            reuse = float(edge_row["reuse_distance"])

    if geo is None:
        geo = float(np.linalg.norm(src_row[["x", "y"]].values.astype(float) - tgt_row[["x", "y"]].values.astype(float)))

    if hyd is None:
        hyd_cols = [c for c in src_row.index if str(c).startswith("hyd_")] + [c for c in src_row.index if str(c).startswith("geo_")]
        if hyd_cols:
            hyd = float(np.linalg.norm(src_row[hyd_cols].values.astype(float) - tgt_row[hyd_cols].values.astype(float)) / max(len(hyd_cols), 1))
        else:
            hyd = geo

    if chem is None:
        chem_cols = [c for c in src_row.index if str(c).startswith("chem_")]
        if chem_cols:
            chem = float(np.linalg.norm(src_row[chem_cols].values.astype(float) - tgt_row[chem_cols].values.astype(float)) / max(len(chem_cols), 1))
        else:
            chem = 0.0

    if reuse is None:
        src_type = str(src_row["source_type"])
        tgt_type = str(tgt_row["source_type"])
        reuse = 0.0 if src_type == tgt_type else 1.0
        if {"reuse", "cross"} & {src_type, tgt_type}:
            reuse *= 0.5

    dissimilarity = (
        weights.get("geographic", 0.25) * geo
        + weights.get("hydro", 0.25) * hyd
        + weights.get("chemical", 0.25) * chem
        + weights.get("reuse", 0.25) * reuse
    )
    affinity = math.exp(-float(dissimilarity) / (1.0 + 1e-8))
    return float(affinity)


def infer_edges_if_missing(data: DataBundle, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a simple default edge list when an explicit edges CSV is absent.

    The inferred network mixes:
        - within-source k-nearest neighbors,
        - cross-source nearest connectors involving the 'cross' nodes,
        - limited nearest relations between reuse and groundwater / surface
          to represent coupled contamination opportunities.
    """
    df = data.nodes_df
    k = int(cfg["graph"].get("k_nearest_if_edges_missing", 5))
    rows = []
    source_types = df["source_type"].astype(str).unique().tolist()

    # Within-source relations.
    for source_type in source_types:
        sub = df[df["source_type"].astype(str) == source_type].copy()
        if len(sub) <= 1:
            continue
        coords = sub[["x", "y"]].values
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(sub))).fit(coords)
        dists, idxs = nbrs.kneighbors(coords)
        relation_type = "surface" if source_type == "surface" else (
            "groundwater" if source_type == "groundwater" else (
                "reuse" if source_type == "reuse" else "cross"
            )
        )
        for i, src_id in enumerate(sub["node_id"].tolist()):
            for d, j in zip(dists[i, 1:], idxs[i, 1:]):
                tgt_id = sub.iloc[j]["node_id"]
                rows.append(
                    {
                        "source": src_id,
                        "target": tgt_id,
                        "relation_type": relation_type,
                        "geographic_distance": float(d),
                    }
                )

    # Cross-source relations. These are intentionally sparse to avoid overconnecting the graph.
    combos = [("surface", "groundwater"), ("surface", "reuse"), ("groundwater", "reuse"), ("cross", "surface"), ("cross", "groundwater"), ("cross", "reuse")]
    for a, b in combos:
        sub_a = df[df["source_type"].astype(str) == a].copy()
        sub_b = df[df["source_type"].astype(str) == b].copy()
        if len(sub_a) == 0 or len(sub_b) == 0:
            continue
        nbrs = NearestNeighbors(n_neighbors=min(2, len(sub_b))).fit(sub_b[["x", "y"]].values)
        dists, idxs = nbrs.kneighbors(sub_a[["x", "y"]].values)
        for i, src_id in enumerate(sub_a["node_id"].tolist()):
            for d, j in zip(dists[i], idxs[i]):
                tgt_id = sub_b.iloc[j]["node_id"]
                rows.append(
                    {
                        "source": src_id,
                        "target": tgt_id,
                        "relation_type": "cross" if "cross" in [a, b] else "reuse",
                        "geographic_distance": float(d),
                    }
                )

    edges_df = pd.DataFrame(rows).drop_duplicates(subset=["source", "target", "relation_type"])
    logging.info("Inferred %d edges because no edge file was found.", len(edges_df))
    return edges_df


def build_graph_bundle(data: DataBundle, cfg: Dict[str, Any]) -> GraphBundle:
    """
    Build the multiplex graph, one directed graph per relation type,
    plus an aggregate graph and adjacency matrices for learning.

    Relation types expected in this study:
        - surface
        - groundwater
        - reuse
        - cross
    """
    edges_path = Path(cfg["data"]["edges_csv"])
    if not edges_path.exists():
        if cfg["project"].get("generate_demo_data_if_missing", True):
            generate_demo_dataset(Path(cfg["data"]["nodes_csv"]), edges_path, cfg["project"]["seed"])
        else:
            edges_df = infer_edges_if_missing(data, cfg)
    if edges_path.exists():
        edges_df = pd.read_csv(edges_path)

    if "relation_type" not in edges_df.columns:
        edges_df["relation_type"] = "cross"

    # Guarantee standard column names used internally by the graph builder.
    rename_map = {}
    if cfg["data"].get("source_col", "source") != "source":
        rename_map[cfg["data"].get("source_col")] = "source"
    if cfg["data"].get("target_col", "target") != "target":
        rename_map[cfg["data"].get("target_col")] = "target"
    if cfg["data"].get("relation_type_col", "relation_type") != "relation_type":
        rename_map[cfg["data"].get("relation_type_col")] = "relation_type"
    if rename_map:
        edges_df = edges_df.rename(columns=rename_map)

    required = ["source", "target", "relation_type"]
    for col in required:
        if col not in edges_df.columns:
            raise ValueError(f"Required edge column '{col}' not found.")

    relation_names = sorted(edges_df["relation_type"].astype(str).unique().tolist())
    node_index = {node_id: i for i, node_id in enumerate(data.node_ids)}
    node_row_lookup = data.nodes_df.set_index("node_id", drop=False)

    relation_graphs: Dict[str, nx.DiGraph] = {}
    for rel in relation_names:
        G = nx.DiGraph()
        G.add_nodes_from(data.node_ids)
        relation_graphs[rel] = G

    aggregate_graph = nx.DiGraph()
    aggregate_graph.add_nodes_from(data.node_ids)

    # Compute or normalize edge weights while dropping any invalid node references.
    valid_rows = []
    for _, edge in edges_df.iterrows():
        src = str(edge["source"])
        tgt = str(edge["target"])
        if src not in node_row_lookup.index or tgt not in node_row_lookup.index:
            continue

        src_row = node_row_lookup.loc[src]
        tgt_row = node_row_lookup.loc[tgt]

        if "weight" in edge.index and not pd.isna(edge["weight"]):
            weight = float(edge["weight"])
        else:
            weight = compute_edge_weight(src_row, tgt_row, edge, cfg)

        row_dict = edge.to_dict()
        row_dict["weight"] = weight
        valid_rows.append(row_dict)

    edges_df = pd.DataFrame(valid_rows)

    if cfg["graph"].get("normalize_edge_weights", True):
        w = edges_df["weight"].values
        edges_df["weight"] = normalize_minmax(w) + 1e-4

    for _, edge in edges_df.iterrows():
        src = str(edge["source"])
        tgt = str(edge["target"])
        rel = str(edge["relation_type"])
        weight = float(edge["weight"])
        if rel not in relation_graphs:
            relation_graphs[rel] = nx.DiGraph()
            relation_graphs[rel].add_nodes_from(data.node_ids)
            relation_names.append(rel)
        relation_graphs[rel].add_edge(src, tgt, weight=weight)
        aggregate_graph.add_edge(src, tgt, weight=aggregate_graph.get_edge_data(src, tgt, {}).get("weight", 0.0) + weight)

    relation_names = sorted(list(set(relation_names)))

    # Dense adjacency matrices keep the code easy to understand and inspect.
    adjacency_by_relation = {}
    n = len(data.node_ids)
    for rel in relation_names:
        A = np.zeros((n, n), dtype=np.float32)
        G = relation_graphs[rel]
        for u, v, attrs in G.edges(data=True):
            A[node_index[u], node_index[v]] = float(attrs.get("weight", 1.0))
        adjacency_by_relation[rel] = row_normalize(A)

    A_agg = np.zeros((n, n), dtype=np.float32)
    for u, v, attrs in aggregate_graph.edges(data=True):
        A_agg[node_index[u], node_index[v]] = float(attrs.get("weight", 1.0))
    A_agg = row_normalize(A_agg)

    # Graph statistics for later ablation and interpretation.
    UG = aggregate_graph.to_undirected()
    degree_dict = dict(UG.degree())
    strength_dict = dict(UG.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(UG, weight="weight")
    closeness = nx.closeness_centrality(UG)
    pagerank = nx.pagerank(aggregate_graph, weight="weight")
    clustering = nx.clustering(UG, weight="weight")

    graph_stats = pd.DataFrame(
        {
            "node_id": data.node_ids,
            "degree": [degree_dict.get(nid, 0.0) for nid in data.node_ids],
            "strength": [strength_dict.get(nid, 0.0) for nid in data.node_ids],
            "betweenness": [betweenness.get(nid, 0.0) for nid in data.node_ids],
            "closeness": [closeness.get(nid, 0.0) for nid in data.node_ids],
            "pagerank": [pagerank.get(nid, 0.0) for nid in data.node_ids],
            "clustering": [clustering.get(nid, 0.0) for nid in data.node_ids],
        }
    )

    logging.info(
        "Graph bundle built with %d nodes, %d aggregate edges, and relations=%s",
        len(data.node_ids),
        aggregate_graph.number_of_edges(),
        relation_names,
    )

    return GraphBundle(
        relation_graphs=relation_graphs,
        aggregate_graph=aggregate_graph,
        adjacency_by_relation=adjacency_by_relation,
        aggregate_adjacency=A_agg,
        graph_stats=graph_stats,
        relation_names=relation_names,
        edge_df=edges_df,
    )


# --------------------------------------------------------------------------------------
# Topological feature extraction
# --------------------------------------------------------------------------------------

def neighborhood_nodes(aggregate_graph: nx.DiGraph, center: str, hops: int = 2) -> List[str]:
    """
    Extract a local or mesoscopic neighborhood around a node.
    The graph is converted to undirected mode because local topological structure
    is usually interpreted in a connectivity sense rather than in a purely flow-directed sense.
    """
    UG = aggregate_graph.to_undirected()
    lengths = nx.single_source_shortest_path_length(UG, center, cutoff=hops)
    return sorted(list(lengths.keys()))


def local_distance_matrix(local_df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Build the local dissimilarity matrix used by persistent homology.

    We combine scaled analytical features and coordinates because pollution risk
    in urban water systems depends simultaneously on environmental attributes
    and spatial arrangement.
    """
    cols = feature_cols + ["x", "y"]
    X = local_df[cols].copy()
    # Standard scaling inside each neighborhood avoids one very large feature block
    # dominating the filtration.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))
    D = squareform(pdist(X_scaled, metric="euclidean"))
    return D.astype(np.float64)


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """
    Remove infinite deaths because persistence images and finite-scale summaries
    need bounded support.
    """
    if diagram is None or len(diagram) == 0:
        return np.zeros((0, 2), dtype=float)
    diagram = np.asarray(diagram, dtype=float)
    mask = np.isfinite(diagram[:, 1])
    return diagram[mask]


def betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Compute a simple Betti curve from a persistence diagram by counting
    how many intervals are alive at each filtration value.
    """
    dgm = finite_diagram(diagram)
    if len(dgm) == 0:
        return np.zeros(len(grid), dtype=np.float32)
    counts = np.zeros(len(grid), dtype=np.float32)
    for birth, death in dgm:
        counts += ((grid >= birth) & (grid <= death)).astype(np.float32)
    return counts


def diagram_ranges(diagrams: List[np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Determine the common birth and persistence ranges used to create a consistent
    vectorization across neighborhoods.
    """
    births, persistences = [], []
    for dgm in diagrams:
        dgm = finite_diagram(dgm)
        if len(dgm) == 0:
            continue
        births.extend(dgm[:, 0].tolist())
        persistences.extend((dgm[:, 1] - dgm[:, 0]).tolist())

    if len(births) == 0:
        return (0.0, 1.0), (0.0, 1.0)

    bmin, bmax = float(np.min(births)), float(np.max(births))
    pmin, pmax = 0.0, float(np.max(persistences))
    if math.isclose(bmin, bmax):
        bmax += 1.0
    if math.isclose(pmin, pmax):
        pmax += 1.0
    return (bmin, bmax), (pmin, pmax)


def custom_persistence_image(
    diagram: np.ndarray,
    pixel_size: Tuple[int, int],
    birth_range: Tuple[float, float],
    pers_range: Tuple[float, float],
    sigma: float = 0.12,
) -> np.ndarray:
    """
    Minimal persistence-image implementation used when Persim is unavailable or
    when the user wants a transparent fallback implementation.

    The diagram is converted from birth-death to birth-persistence coordinates,
    then each point contributes a Gaussian kernel weighted by persistence.
    """
    dgm = finite_diagram(diagram)
    image = np.zeros(pixel_size, dtype=np.float32)
    if len(dgm) == 0:
        return image

    bx = np.linspace(birth_range[0], birth_range[1], pixel_size[0])
    py = np.linspace(pers_range[0], pers_range[1], pixel_size[1])
    XX, YY = np.meshgrid(bx, py, indexing="ij")

    for birth, death in dgm:
        persistence = max(float(death - birth), 0.0)
        image += persistence * np.exp(-((XX - birth) ** 2 + (YY - persistence) ** 2) / (2 * sigma ** 2))

    return image.astype(np.float32)


def resample_2d_array(array_2d: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D array to a fixed target shape using deterministic bilinear-style
    interpolation implemented with NumPy only.

    Why this helper is necessary:
        Persim's PersistenceImager does not interpret ``pixel_size`` as an output
        matrix shape. Instead, it interprets it as a physical pixel resolution on
        the birth-persistence plane. Consequently, the matrix produced by
        ``transform`` may be much larger than the user-configured analytical grid.

        A second edge case occurs when a persistence diagram is empty. In the
        original code, the empty case produced ``np.zeros(pixel_size)`` while the
        non-empty case returned Persim's native output shape. Concatenating those
        mixed vectors caused ``np.vstack`` to fail because some nodes had shorter
        topological vectors.

    This function eliminates that instability by coercing every persistence image
    to the same fixed shape defined in the configuration file.
    """
    arr = np.asarray(array_2d, dtype=np.float32)

    if arr.ndim == 0:
        return np.full(target_shape, float(arr), dtype=np.float32)

    if arr.ndim == 1:
        target_len = int(target_shape[0] * target_shape[1])
        if arr.size == target_len:
            return arr.reshape(target_shape).astype(np.float32)
        x_old = np.linspace(0.0, 1.0, num=max(arr.size, 2), dtype=np.float32)
        if arr.size == 1:
            arr = np.repeat(arr, 2)
        x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
        resized = np.interp(x_new, x_old, arr).reshape(target_shape)
        return resized.astype(np.float32)

    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim != 2:
        arr = np.atleast_2d(arr).astype(np.float32)

    if arr.shape == target_shape:
        return arr.astype(np.float32)

    old_x = np.linspace(0.0, 1.0, num=max(arr.shape[0], 2), dtype=np.float32)
    old_y = np.linspace(0.0, 1.0, num=max(arr.shape[1], 2), dtype=np.float32)

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 2, axis=0)
        old_x = np.linspace(0.0, 1.0, num=2, dtype=np.float32)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
        old_y = np.linspace(0.0, 1.0, num=2, dtype=np.float32)

    new_x = np.linspace(0.0, 1.0, num=target_shape[0], dtype=np.float32)
    new_y = np.linspace(0.0, 1.0, num=target_shape[1], dtype=np.float32)

    temp = np.empty((target_shape[0], arr.shape[1]), dtype=np.float32)
    for j in range(arr.shape[1]):
        temp[:, j] = np.interp(new_x, old_x, arr[:, j])

    resized = np.empty(target_shape, dtype=np.float32)
    for i in range(target_shape[0]):
        resized[i, :] = np.interp(new_y, old_y, temp[i, :])

    return resized.astype(np.float32)


def coerce_persistence_image_shape(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert any persistence-image-like output to the exact configured shape.

    The helper accepts:
        - native Persim 2D outputs,
        - already well-formed custom outputs,
        - 1D flattened vectors,
        - scalar edge cases,
        - squeezed higher-dimensional arrays from backend/version differences.

    The result is always a float32 matrix with shape ``target_shape``.
    """
    if image is None:
        return np.zeros(target_shape, dtype=np.float32)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return resample_2d_array(arr, target_shape)


def manual_hodge_1_laplacian_stats(UG: nx.Graph) -> np.ndarray:
    """
    Compute a small set of Hodge-inspired statistics from the 1-Laplacian
    of the local clique complex up to dimension 2.

    This manual implementation makes the pipeline robust even when TopoNetX
    is not available or when its API differs across versions.

    Steps:
        1. Build an oriented edge list.
        2. Enumerate triangles as 2-simplices.
        3. Assemble B1 and B2 incidence matrices.
        4. Compute L1 = B1^T B1 + B2 B2^T.
        5. Extract spectral summary statistics.
    """
    nodes = sorted(UG.nodes())
    if len(nodes) < 3 or UG.number_of_edges() < 2:
        return np.zeros(6, dtype=np.float32)

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = sorted({tuple(sorted((u, v))) for u, v in UG.edges()})
    edge_to_idx = {e: i for i, e in enumerate(edges)}

    if len(edges) == 0:
        return np.zeros(6, dtype=np.float32)

    triangles = []
    for clique in nx.enumerate_all_cliques(UG):
        if len(clique) == 3:
            triangles.append(tuple(sorted(clique)))
        elif len(clique) > 3:
            break

    B1 = np.zeros((len(nodes), len(edges)), dtype=np.float64)
    for e_idx, (u, v) in enumerate(edges):
        # Orientation is induced by sorted node order.
        B1[node_to_idx[u], e_idx] = -1.0
        B1[node_to_idx[v], e_idx] = 1.0

    B2 = np.zeros((len(edges), len(triangles)), dtype=np.float64)
    for t_idx, tri in enumerate(triangles):
        a, b, c = tri
        oriented_edges = [((b, c), 1.0), ((a, c), -1.0), ((a, b), 1.0)]
        for edge, sign in oriented_edges:
            edge = tuple(sorted(edge))
            if edge in edge_to_idx:
                B2[edge_to_idx[edge], t_idx] = sign

    L1 = B1.T @ B1
    if len(triangles) > 0:
        L1 = L1 + B2 @ B2.T

    eigvals = np.linalg.eigvalsh(L1)
    eigvals = np.real(eigvals)
    eigvals[eigvals < 0] = 0.0

    positive = eigvals[eigvals > 1e-8]
    zero_count = float(np.sum(eigvals <= 1e-8))
    spectral_entropy = float(entropy((positive / positive.sum()) if positive.sum() > 0 else np.ones(1)))

    stats = np.array(
        [
            float(np.mean(eigvals)),
            float(np.std(eigvals)),
            float(np.min(positive)) if len(positive) > 0 else 0.0,
            float(np.max(eigvals)) if len(eigvals) > 0 else 0.0,
            zero_count,
            spectral_entropy,
        ],
        dtype=np.float32,
    )
    return stats


def compute_topological_embeddings(data: DataBundle, graph: GraphBundle, cfg: Dict[str, Any]) -> TopologyBundle:
    """
    Compute local topological descriptors for each node:
        - persistence images for H0 and H1,
        - Betti curves for H0 and H1,
        - Hodge-inspired spectral descriptors.

    The design follows the article's methodology:
        topological_embedding = [persistence_image, Betti summaries, Hodge stats]

    The code caches neighborhoods because many adjacent nodes share the same local support.
    """
    ensure_topology_dependencies()

    topo_cfg = cfg["topology"]
    hops = int(topo_cfg.get("neighborhood_hops", 2))
    maxdim = int(topo_cfg.get("max_persistence_dimension", 1))
    betti_grid_size = int(topo_cfg.get("betti_grid_size", 32))
    pixels = topo_cfg.get("persistence_pixels", [12, 12])
    pixel_size = (int(pixels[0]), int(pixels[1]))

    feature_cols = data.feature_names
    node_lookup = data.nodes_df.set_index("node_id", drop=False)

    diagrams: Dict[str, Dict[int, np.ndarray]] = {}
    betti_curves_dict: Dict[str, Dict[int, np.ndarray]] = {}
    persistence_images_dict: Dict[str, Dict[int, np.ndarray]] = {}
    hodge_stats_dict: Dict[str, np.ndarray] = {}
    neighborhoods_dict: Dict[str, List[str]] = {}

    # First pass: compute diagrams and local Hodge statistics.
    h0_diagrams, h1_diagrams = [], []
    neighborhood_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for node_id in data.node_ids:
        neigh = neighborhood_nodes(graph.aggregate_graph, node_id, hops=hops)
        neighborhoods_dict[node_id] = neigh
        cache_key = tuple(neigh)

        if cache_key in neighborhood_cache:
            cached = neighborhood_cache[cache_key]
            diagrams[node_id] = cached["diagrams"]
            hodge_stats_dict[node_id] = cached["hodge_stats"]
            continue

        local_df = node_lookup.loc[list(neigh)].copy()
        D = local_distance_matrix(local_df, feature_cols)
        ripser_result = ripser(D, distance_matrix=True, maxdim=maxdim)
        dgms = ripser_result["dgms"]

        # Protect against missing H1 when the neighborhood is too small.
        diagrams_local = {
            0: np.asarray(dgms[0], dtype=float) if len(dgms) >= 1 else np.zeros((0, 2)),
            1: np.asarray(dgms[1], dtype=float) if len(dgms) >= 2 else np.zeros((0, 2)),
        }

        h0_diagrams.append(diagrams_local[0])
        h1_diagrams.append(diagrams_local[1])

        UG = graph.aggregate_graph.subgraph(neigh).to_undirected()
        hodge_stats = manual_hodge_1_laplacian_stats(UG)

        diagrams[node_id] = diagrams_local
        hodge_stats_dict[node_id] = hodge_stats
        neighborhood_cache[cache_key] = {
            "diagrams": diagrams_local,
            "hodge_stats": hodge_stats,
        }

    # Create a common grid for Betti curves and a common persistence-image domain.
    all_dgms = h0_diagrams + h1_diagrams
    birth_range, pers_range = diagram_ranges(all_dgms)
    max_filtration = max(birth_range[1], pers_range[1])
    grid = np.linspace(0.0, max_filtration, betti_grid_size)

    # Fit Persim when available; otherwise use the transparent custom implementation.
    # Important implementation note:
    #     Persim's `pixel_size` argument is not the final matrix shape; it is the
    #     physical resolution on the birth-persistence plane. Therefore, the raw
    #     output shape may differ from the configuration value `persistence_pixels`.
    #     To keep the analytical feature space stable, every image is later coerced
    #     to `pixel_size` with `coerce_persistence_image_shape`.
    use_persim = PersistenceImager is not None
    pimgr = None
    non_empty_fit_diagrams = [finite_diagram(d) for d in all_dgms if len(finite_diagram(d)) > 0]
    if use_persim and len(non_empty_fit_diagrams) > 0:
        pimgr = PersistenceImager(pixel_size=0.1)
        try:
            pimgr.fit(non_empty_fit_diagrams, skew=True)
        except Exception:
            # Some versions of Persim behave differently. The fallback remains valid.
            use_persim = False
            pimgr = None
    else:
        use_persim = False

    topo_vectors = []
    topo_feature_names = []

    # Construct reproducible feature names.
    n_pi = pixel_size[0] * pixel_size[1]
    for prefix in ["pi_h0", "pi_h1"]:
        for i in range(n_pi):
            topo_feature_names.append(f"{prefix}_{i:03d}")
    for prefix in ["betti0", "betti1"]:
        for i in range(betti_grid_size):
            topo_feature_names.append(f"{prefix}_{i:03d}")
    topo_feature_names.extend([
        "hodge_mean_eig",
        "hodge_std_eig",
        "hodge_min_pos_eig",
        "hodge_max_eig",
        "hodge_zero_count",
        "hodge_spectral_entropy",
    ])

    expected_vector_len = int(2 * n_pi + 2 * betti_grid_size + 6)

    for node_id in data.node_ids:
        node_diagrams = diagrams[node_id]

        betti0 = betti_curve(node_diagrams[0], grid)
        betti1 = betti_curve(node_diagrams[1], grid)
        betti_curves_dict[node_id] = {0: betti0, 1: betti1}

        if use_persim and pimgr is not None:
            try:
                raw_h0 = pimgr.transform(finite_diagram(node_diagrams[0]), skew=True) if len(finite_diagram(node_diagrams[0])) > 0 else np.zeros(pixel_size, dtype=np.float32)
                raw_h1 = pimgr.transform(finite_diagram(node_diagrams[1]), skew=True) if len(finite_diagram(node_diagrams[1])) > 0 else np.zeros(pixel_size, dtype=np.float32)
                pi_h0 = coerce_persistence_image_shape(raw_h0, pixel_size)
                pi_h1 = coerce_persistence_image_shape(raw_h1, pixel_size)
            except Exception:
                pi_h0 = custom_persistence_image(node_diagrams[0], pixel_size, birth_range, pers_range)
                pi_h1 = custom_persistence_image(node_diagrams[1], pixel_size, birth_range, pers_range)
        else:
            pi_h0 = custom_persistence_image(node_diagrams[0], pixel_size, birth_range, pers_range)
            pi_h1 = custom_persistence_image(node_diagrams[1], pixel_size, birth_range, pers_range)

        # A second safeguard keeps the representation stable even if a backend
        # returns a different shape due to version-specific behavior.
        pi_h0 = coerce_persistence_image_shape(pi_h0, pixel_size)
        pi_h1 = coerce_persistence_image_shape(pi_h1, pixel_size)

        persistence_images_dict[node_id] = {0: pi_h0, 1: pi_h1}

        vector = np.concatenate(
            [
                pi_h0.flatten(),
                pi_h1.flatten(),
                betti0.astype(np.float32),
                betti1.astype(np.float32),
                hodge_stats_dict[node_id].astype(np.float32),
            ]
        ).astype(np.float32)

        if vector.size != expected_vector_len:
            raise ValueError(
                f"Node {node_id} produced a topological vector of length {vector.size}, "
                f"but the expected length is {expected_vector_len}. "
                f"This indicates an upstream shape inconsistency."
            )

        topo_vectors.append(vector)

    topo_matrix = np.vstack(topo_vectors).astype(np.float32)

    # Representative nodes are selected by label severity and source diversity.
    topo_norm = np.linalg.norm(topo_matrix, axis=1)
    ranking = np.argsort(-(normalize_minmax(topo_norm) + 0.5 * data.labels))
    representative_nodes = [data.node_ids[i] for i in ranking[: int(topo_cfg.get("representative_nodes_top_k", 12))]]

    logging.info("Topological embeddings computed with shape %s", topo_matrix.shape)

    return TopologyBundle(
        topo_matrix=topo_matrix,
        topo_feature_names=topo_feature_names,
        diagrams=diagrams,
        betti_curves=betti_curves_dict,
        persistence_images=persistence_images_dict,
        hodge_stats=hodge_stats_dict,
        neighborhoods=neighborhoods_dict,
        representative_nodes=representative_nodes,
    )


# --------------------------------------------------------------------------------------
# Learning model
# --------------------------------------------------------------------------------------

class TopologyAwareLayer(nn.Module):
    """
    One message-passing layer with:
        - a self-update term,
        - relation-specific neighborhood aggregation,
        - explicit topological injection at every layer,
        - normalization and dropout.

    This directly mirrors the methodology in Section 3.2, where the topological
    embedding is injected into each hidden layer so it is not washed out by repeated
    aggregation.
    """
    def __init__(self, hidden_dim: int, topo_dim: int, relation_names: List[str], dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.topo_linear = nn.Linear(topo_dim, hidden_dim, bias=False)
        self.relation_linears = nn.ModuleDict({
            rel: nn.Linear(hidden_dim, hidden_dim, bias=False) for rel in relation_names
        })
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, topo: torch.Tensor, adjacency_by_relation: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.self_linear(h) + self.topo_linear(topo)
        for rel, A in adjacency_by_relation.items():
            out = out + self.relation_linears[rel](A @ h)
        out = self.norm(F.relu(out))
        out = self.dropout(out)
        return out


class UrbanWaterTDL(nn.Module):
    """
    Complete topology-aware network used for pollution-risk inference.
    The model fuses:
        - node features,
        - source-type embeddings,
        - topological descriptors,
        - relation-specific message passing.
    """
    def __init__(
        self,
        input_dim: int,
        topo_dim: int,
        num_source_types: int,
        relation_names: List[str],
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        source_embed_dim: int = 8,
    ):
        super().__init__()
        self.source_embedding = nn.Embedding(num_source_types, source_embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + source_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([
            TopologyAwareLayer(hidden_dim, topo_dim, relation_names, dropout)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + topo_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        topo: torch.Tensor,
        source_ids: torch.Tensor,
        adjacency_by_relation: Dict[str, torch.Tensor],
        return_embeddings: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        src_emb = self.source_embedding(source_ids)
        h = self.encoder(torch.cat([x, src_emb], dim=1))
        for layer in self.layers:
            h = layer(h, topo, adjacency_by_relation)
        logits = self.output_head(torch.cat([h, topo], dim=1)).squeeze(-1)
        if return_embeddings:
            return logits, h
        return logits


def build_topology_similarity_matrix(topo_matrix: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Construct a similarity matrix in topological-embedding space.
    This supports the topology-consistency regularizer.
    """
    if len(topo_matrix) <= 1:
        return np.zeros((len(topo_matrix), len(topo_matrix)), dtype=np.float32)

    n_neighbors = min(k + 1, len(topo_matrix))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(topo_matrix)
    dists, idxs = nbrs.kneighbors(topo_matrix)

    sigma = np.median(dists[:, 1:]) if dists.shape[1] > 1 else 1.0
    sigma = max(float(sigma), 1e-6)

    S = np.zeros((len(topo_matrix), len(topo_matrix)), dtype=np.float32)
    for i in range(len(topo_matrix)):
        for d, j in zip(dists[i, 1:], idxs[i, 1:]):
            sim = math.exp(-float(d) ** 2 / (2 * sigma ** 2))
            S[i, j] = max(S[i, j], sim)
            S[j, i] = max(S[j, i], sim)
    return row_normalize(S)


def class_weighted_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Class-weighted BCE to mitigate the usual rarity of critical contamination events.
    """
    y = labels.detach().cpu().numpy()
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = max(neg / (pos + 1e-8), 1.0)
    weights = torch.where(labels > 0.5, torch.tensor(pos_weight, device=logits.device), torch.tensor(1.0, device=logits.device))
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights)


def composite_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topo_similarity: torch.Tensor,
    aggregate_adjacency: torch.Tensor,
    train_mask: torch.Tensor,
    topo_weight: float,
    calibration_weight: float,
    smoothness_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Composite training objective:
        - supervised class-weighted BCE,
        - topology-consistency regularizer,
        - probabilistic calibration via Brier score,
        - graph smoothness over strongly coupled nodes.
    """
    train_logits = logits[train_mask]
    train_labels = labels[train_mask]

    bce = class_weighted_bce(train_logits, train_labels)
    probs = torch.sigmoid(logits)

    # Topology consistency is computed over the full graph because it is a structural regularizer.
    prob_diffs = probs.unsqueeze(1) - probs.unsqueeze(0)
    topo_consistency = (topo_similarity * prob_diffs.pow(2)).sum() / (topo_similarity.sum() + 1e-8)

    # Brier score on the supervised subset encourages calibrated probabilities.
    calibration = torch.mean((torch.sigmoid(train_logits) - train_labels) ** 2)

    # Smoothness discourages implausible discontinuities across strong aggregate relations.
    smooth_diffs = probs.unsqueeze(1) - probs.unsqueeze(0)
    smoothness = (aggregate_adjacency * smooth_diffs.pow(2)).sum() / (aggregate_adjacency.sum() + 1e-8)

    total = bce + topo_weight * topo_consistency + calibration_weight * calibration + smoothness_weight * smoothness

    return total, {
        "bce": float(bce.detach().cpu()),
        "topo_consistency": float(topo_consistency.detach().cpu()),
        "calibration": float(calibration.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def make_masks_from_indices(n: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform index arrays into boolean masks."""
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.tensor(train_idx)] = True
    val_mask[torch.tensor(val_idx)] = True
    test_mask[torch.tensor(test_idx)] = True
    return train_mask, val_mask, test_mask


def create_spatial_split(data: DataBundle, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation/test splits that are spatially or administratively aware.
    This reduces leakage and aligns evaluation with realistic deployment conditions.

    Priority:
        1. If group labels (districts, sectors, etc.) are informative, use them.
        2. Otherwise generate KMeans spatial blocks from coordinates.
    """
    eval_cfg = cfg["evaluation"]
    groups = np.asarray(data.groups)

    # If there is effectively only one group, create spatial blocks from coordinates.
    if len(np.unique(groups)) <= 1:
        n_blocks = int(eval_cfg.get("n_spatial_blocks", 8))
        km_blocks = KMeans(n_clusters=min(n_blocks, len(data.node_ids)), random_state=cfg["project"]["seed"], n_init=10)
        groups = km_blocks.fit_predict(data.coords).astype(str)

    indices = np.arange(len(data.node_ids))
    val_fraction = float(eval_cfg.get("val_fraction", 0.15))
    test_fraction = float(eval_cfg.get("test_fraction", 0.20))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=cfg["project"]["seed"])
    train_val_idx, test_idx = next(gss_test.split(indices, data.labels, groups))

    remaining_groups = groups[train_val_idx]
    # Relative validation fraction within the remaining subset.
    rel_val = val_fraction / max(1.0 - test_fraction, 1e-8)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=cfg["project"]["seed"] + 1)
    inner_train_idx, inner_val_idx = next(gss_val.split(train_val_idx, data.labels[train_val_idx], remaining_groups))

    train_idx = train_val_idx[inner_train_idx]
    val_idx = train_val_idx[inner_val_idx]

    logging.info(
        "Spatial split created: train=%d, val=%d, test=%d",
        len(train_idx), len(val_idx), len(test_idx)
    )
    return train_idx, val_idx, test_idx


def train_model(
    data: DataBundle,
    graph: GraphBundle,
    topo: np.ndarray,
    cfg: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    run_name: str = "full_model",
) -> TrainingArtifacts:
    """
    Train the topology-aware model and return both predictions and latent embeddings.
    """
    model_cfg = cfg["model"]
    device = default_device()

    X = torch.tensor(data.feature_matrix, dtype=torch.float32, device=device)
    T = torch.tensor(topo, dtype=torch.float32, device=device)
    y = torch.tensor(data.labels.astype(np.float32), dtype=torch.float32, device=device)
    s = torch.tensor(data.source_type_ids.astype(np.int64), dtype=torch.long, device=device)

    A_rel = {
        rel: torch.tensor(adj, dtype=torch.float32, device=device)
        for rel, adj in graph.adjacency_by_relation.items()
    }
    A_agg = torch.tensor(graph.aggregate_adjacency, dtype=torch.float32, device=device)
    topo_sim = torch.tensor(
        build_topology_similarity_matrix(topo, k=int(model_cfg.get("topo_knn", 8))),
        dtype=torch.float32,
        device=device,
    )

    train_mask, val_mask, test_mask = make_masks_from_indices(len(data.node_ids), train_idx, val_idx, test_idx)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    model = UrbanWaterTDL(
        input_dim=data.feature_matrix.shape[1],
        topo_dim=topo.shape[1],
        num_source_types=len(data.source_type_names),
        relation_names=graph.relation_names,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        source_embed_dim=int(model_cfg.get("source_embed_dim", 8)),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )

    best_state = None
    best_val_score = -np.inf
    patience = int(model_cfg.get("patience", 20))
    epochs = int(model_cfg.get("epochs", 150))
    no_improve = 0
    history_rows = []

    topo_loss_weight = float(model_cfg.get("topo_loss_weight", 0.4))
    calibration_loss_weight = float(model_cfg.get("calibration_loss_weight", 0.2))
    smoothness_loss_weight = float(model_cfg.get("smoothness_loss_weight", 0.1))

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X, T, s, A_rel)
        loss, loss_dict = composite_loss(
            logits=logits,
            labels=y,
            topo_similarity=topo_sim,
            aggregate_adjacency=A_agg,
            train_mask=train_mask,
            topo_weight=topo_loss_weight,
            calibration_weight=calibration_loss_weight,
            smoothness_weight=smoothness_loss_weight,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval, embeddings = model(X, T, s, A_rel, return_embeddings=True)
            probs = torch.sigmoid(logits_eval).detach().cpu().numpy()
            val_probs = probs[val_idx]
            val_y = data.labels[val_idx]

            if len(np.unique(val_y)) > 1:
                val_score = average_precision_score(val_y, val_probs)
            else:
                val_score = 0.0

        row = {"epoch": epoch, "val_average_precision": val_score, **loss_dict}
        history_rows.append(row)

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logging.info("[%s] Early stopping at epoch %d", run_name, epoch)
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_logits, final_embeddings = model(X, T, s, A_rel, return_embeddings=True)
        final_probs = torch.sigmoid(final_logits).detach().cpu().numpy()
        latent_embeddings = final_embeddings.detach().cpu().numpy()

    threshold = float(cfg["evaluation"].get("classification_threshold", 0.50))
    test_probs = final_probs[test_idx]
    test_pred = (test_probs >= threshold).astype(int)

    history_df = pd.DataFrame(history_rows)
    return TrainingArtifacts(
        model=model,
        history=history_df,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        test_probabilities=test_probs,
        test_predictions=test_pred,
        full_probabilities=final_probs,
        latent_embeddings=latent_embeddings,
        threshold=threshold,
    )


# --------------------------------------------------------------------------------------
# Evaluation, ablation, and explainability
# --------------------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute the main predictive metrics for Section 5.1."""
    pred = (probs >= threshold).astype(int)
    metrics = {
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier_score": float(brier_score_loss(y_true, probs)),
    }
    return metrics


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    topo: torch.Tensor,
    source_ids: torch.Tensor,
    adjacency_by_relation: Dict[str, torch.Tensor],
    node_indices: List[int],
    steps: int = 32,
) -> Dict[int, np.ndarray]:
    """
    Integrated gradients for local feature attribution.

    The implementation is intentionally explicit and readable rather than maximally
    optimized, because the point of explainability is traceability and scientific audit.
    """
    model.eval()
    attributions = {}
    baseline = torch.zeros_like(x)

    for idx in node_indices:
        total_grad = torch.zeros(x.shape[1], device=x.device)
        target_delta = x[idx] - baseline[idx]

        for alpha in np.linspace(0.0, 1.0, steps):
            x_interp = baseline + alpha * (x - baseline)
            x_interp.requires_grad_(True)

            logits = model(x_interp, topo, source_ids, adjacency_by_relation)
            target = torch.sigmoid(logits[idx])
            model.zero_grad(set_to_none=True)
            if x_interp.grad is not None:
                x_interp.grad.zero_()
            target.backward(retain_graph=True)

            total_grad += x_interp.grad[idx].detach()

        ig = (target_delta * total_grad / steps).detach().cpu().numpy()
        attributions[idx] = ig

    return attributions


def topological_attribution(
    model: nn.Module,
    x: torch.Tensor,
    topo: torch.Tensor,
    source_ids: torch.Tensor,
    adjacency_by_relation: Dict[str, torch.Tensor],
) -> np.ndarray:
    """
    Estimate node-level topological attribution by measuring the change in predicted
    risk when the topological embedding is removed.
    """
    model.eval()
    with torch.no_grad():
        full_probs = torch.sigmoid(model(x, topo, source_ids, adjacency_by_relation)).detach().cpu().numpy()
        zero_topo = torch.zeros_like(topo)
        no_topo_probs = torch.sigmoid(model(x, zero_topo, source_ids, adjacency_by_relation)).detach().cpu().numpy()
    return full_probs - no_topo_probs


def build_local_explanation_table(
    data: DataBundle,
    topology_attr: np.ndarray,
    feature_attr: Dict[int, np.ndarray],
    probabilities: np.ndarray,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Summarize the strongest local explanations for the most relevant nodes.
    """
    ranking = np.argsort(-(normalize_minmax(probabilities) + normalize_minmax(topology_attr)))
    rows = []
    for idx in ranking[:top_k]:
        attrib = feature_attr.get(idx)
        if attrib is None:
            continue
        top_feature_idx = np.argsort(-np.abs(attrib))[:5]
        top_features = [f"{data.feature_names[j]} ({attrib[j]:.4f})" for j in top_feature_idx]
        rows.append(
            {
                "node_id": data.node_ids[idx],
                "source_type": data.nodes_df.iloc[idx]["source_type"],
                "district": data.nodes_df.iloc[idx]["district"] if "district" in data.nodes_df.columns else "NA",
                "predicted_risk_probability": float(probabilities[idx]),
                "topological_attribution": float(topology_attr[idx]),
                "top_5_feature_contributors": "; ".join(top_features),
            }
        )
    return pd.DataFrame(rows)


def extract_risk_pathways(
    data: DataBundle,
    graph: GraphBundle,
    probabilities: np.ndarray,
    topology_attr: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Identify candidate risk pathways connecting high-risk nodes across the coupled system.

    The pathway score combines:
        - node risk probability,
        - topological amplification,
        - edge affinity.

    This is an operational proxy for the article's idea of structurally meaningful
    risk trajectories.
    """
    top_k = int(cfg["evaluation"].get("top_k_pathways", 15))
    max_path_length = int(cfg["evaluation"].get("max_path_length", 6))

    G = graph.aggregate_graph.copy()
    node_score = normalize_minmax(probabilities) * (1.0 + normalize_minmax(topology_attr))
    node_score_lookup = {nid: node_score[i] for i, nid in enumerate(data.node_ids)}

    # Convert edge weights into path-search costs. Higher affinity and higher node risk
    # should reduce the cost of a path because such pathways are more plausible and relevant.
    for u, v, attrs in G.edges(data=True):
        edge_aff = float(attrs.get("weight", 1.0))
        risk_boost = (node_score_lookup.get(u, 0.0) + node_score_lookup.get(v, 0.0)) / 2.0
        cost = 1.0 / max(edge_aff * (0.25 + risk_boost), 1e-6)
        attrs["cost"] = cost

    # Candidate origin-destination nodes are high risk and should span multiple source domains.
    high_idx = np.where(probabilities >= np.quantile(probabilities, 0.85))[0]
    high_nodes = [data.node_ids[i] for i in high_idx]
    high_df = data.nodes_df.set_index("node_id").loc[high_nodes].copy()

    rows = []
    pathway_graph = nx.DiGraph()

    # Build source-diverse candidate pairs.
    candidate_pairs = []
    for src in high_nodes:
        for tgt in high_nodes:
            if src == tgt:
                continue
            if high_df.loc[src, "source_type"] != high_df.loc[tgt, "source_type"]:
                candidate_pairs.append((src, tgt))

    for src, tgt in candidate_pairs:
        if src not in G.nodes or tgt not in G.nodes:
            continue
        try:
            path = nx.shortest_path(G, source=src, target=tgt, weight="cost")
        except Exception:
            continue
        if len(path) > max_path_length:
            continue

        edge_weights = []
        path_probs = []
        path_topo = []
        for u, v in zip(path[:-1], path[1:]):
            edge_weights.append(float(G[u][v]["weight"]))
            pathway_graph.add_edge(u, v, weight=float(G[u][v]["weight"]))
        for nid in path:
            i = data.node_ids.index(nid)
            path_probs.append(float(probabilities[i]))
            path_topo.append(float(topology_attr[i]))

        score = (
            float(np.mean(path_probs))
            * float(np.mean(normalize_minmax(np.array(path_topo)) + 1.0))
            * float(np.mean(edge_weights))
        )

        rows.append(
            {
                "origin_node": src,
                "origin_source_type": high_df.loc[src, "source_type"],
                "destination_node": tgt,
                "destination_source_type": high_df.loc[tgt, "source_type"],
                "path_nodes": " -> ".join(path),
                "path_length": int(len(path)),
                "mean_node_risk": float(np.mean(path_probs)),
                "mean_topological_attribution": float(np.mean(path_topo)),
                "mean_edge_affinity": float(np.mean(edge_weights)),
                "pathway_score": float(score),
            }
        )

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "origin_node", "destination_node", "path_nodes", "path_length",
            "mean_node_risk", "mean_topological_attribution", "mean_edge_affinity", "pathway_score"
        ]), pathway_graph

    pathways_df = pd.DataFrame(rows).sort_values("pathway_score", ascending=False).drop_duplicates(subset=["path_nodes"]).head(top_k).reset_index(drop=True)
    return pathways_df, pathway_graph


def compute_intervention_priority_table(
    data: DataBundle,
    probabilities: np.ndarray,
    topology_attr: np.ndarray,
) -> pd.DataFrame:
    """
    Compute the intervention-priority score described in Section 3.2:
        priority = risk * structural importance * exposure / (1 + mitigation)
    """
    risk_norm = normalize_minmax(probabilities)
    topo_norm = normalize_minmax(topology_attr)
    exposure_norm = normalize_minmax(data.exposure)
    mitigation_norm = normalize_minmax(data.mitigation)

    priority = risk_norm * (1.0 + topo_norm) * (0.25 + exposure_norm) / (1.0 + mitigation_norm)

    df = data.nodes_df.copy()
    df["predicted_risk_probability"] = probabilities
    df["topological_attribution"] = topology_attr
    df["priority_score"] = priority
    df["priority_rank"] = (-df["priority_score"]).rank(method="dense").astype(int)
    df = df.sort_values(["priority_score", "predicted_risk_probability"], ascending=False).reset_index(drop=True)
    return df


def run_ablation_suite(
    data: DataBundle,
    graph: GraphBundle,
    topology: TopologyBundle,
    cfg: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Train and evaluate three comparable models:
        1. Full topology-aware model.
        2. No-topology model (topological input zeroed).
        3. Graph-statistics baseline (replace topology with simple graph descriptors).

    This directly supports the article's demand for ablation experiments that show
    whether persistent topology contributes beyond ordinary graph statistics.
    """
    topo_matrix_full = topology.topo_matrix.copy()
    topo_matrix_zero = np.zeros_like(topo_matrix_full)

    graph_stats_features = graph.graph_stats[["degree", "strength", "betweenness", "closeness", "pagerank", "clustering"]].values.astype(np.float32)
    graph_stats_scaled = StandardScaler().fit_transform(graph_stats_features).astype(np.float32)

    runs = {
        "full_topology_model": topo_matrix_full,
        "no_topology_model": topo_matrix_zero,
        "graph_statistics_baseline": graph_stats_scaled,
    }

    rows = []
    full_probs_by_run = {}

    for name, topo_matrix in runs.items():
        art = train_model(data, graph, topo_matrix, cfg, train_idx, val_idx, test_idx, run_name=name)
        metrics = compute_metrics(data.labels[test_idx], art.test_probabilities, art.threshold)
        metrics["run_name"] = name
        rows.append(metrics)
        full_probs_by_run[name] = art.full_probabilities

    return pd.DataFrame(rows), full_probs_by_run


# --------------------------------------------------------------------------------------
# Scenario analysis
# --------------------------------------------------------------------------------------

def apply_scenario_to_data_and_graph(
    data: DataBundle,
    graph: GraphBundle,
    scenario_cfg: Dict[str, Any],
) -> Tuple[DataBundle, GraphBundle]:
    """
    Apply simple multiplicative perturbations to node attributes and/or edge weights.
    This supports the scenario-based decision analysis requested for Section 5.3.
    """
    data_new = copy.deepcopy(data)
    graph_new = copy.deepcopy(graph)

    node_multipliers = scenario_cfg.get("node_multipliers", {})
    if node_multipliers:
        df = data_new.nodes_df.copy()
        for col, mult in node_multipliers.items():
            if col in df.columns:
                df[col] = df[col].astype(float) * float(mult)

        # Rebuild the numerical feature matrix from the updated DataFrame.
        feature_cols = data.feature_names
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        data_new.feature_matrix = scaler.fit_transform(imputer.fit_transform(df[feature_cols])).astype(np.float32)
        data_new.nodes_df = df

    edge_type_multipliers = scenario_cfg.get("edge_type_weight_multipliers", {})
    if edge_type_multipliers:
        for rel, mult in edge_type_multipliers.items():
            if rel in graph_new.relation_graphs:
                G = graph_new.relation_graphs[rel]
                for u, v, attrs in G.edges(data=True):
                    attrs["weight"] = float(attrs.get("weight", 1.0)) * float(mult)

        # Rebuild adjacency matrices and aggregate graph weights.
        n = len(data_new.node_ids)
        node_index = {nid: i for i, nid in enumerate(data_new.node_ids)}
        for rel, G in graph_new.relation_graphs.items():
            A = np.zeros((n, n), dtype=np.float32)
            for u, v, attrs in G.edges(data=True):
                A[node_index[u], node_index[v]] = float(attrs.get("weight", 1.0))
            graph_new.adjacency_by_relation[rel] = row_normalize(A)

        graph_new.aggregate_graph = nx.DiGraph()
        graph_new.aggregate_graph.add_nodes_from(data_new.node_ids)
        for rel, G in graph_new.relation_graphs.items():
            for u, v, attrs in G.edges(data=True):
                weight = float(attrs.get("weight", 1.0))
                current = graph_new.aggregate_graph.get_edge_data(u, v, {}).get("weight", 0.0)
                graph_new.aggregate_graph.add_edge(u, v, weight=current + weight)

        A_agg = np.zeros((n, n), dtype=np.float32)
        for u, v, attrs in graph_new.aggregate_graph.edges(data=True):
            A_agg[node_index[u], node_index[v]] = float(attrs.get("weight", 1.0))
        graph_new.aggregate_adjacency = row_normalize(A_agg)

    return data_new, graph_new


# --------------------------------------------------------------------------------------
# Visualization and export helpers
# --------------------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Export a dataframe to CSV with reproducible formatting."""
    df.to_csv(path, index=False)


def plot_training_history(history: pd.DataFrame, path: Path, title: str) -> None:
    """Plot learning curves without forcing custom colors."""
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["total"], label="Total loss")
    if "val_average_precision" in history.columns:
        plt.plot(history["epoch"], history["val_average_precision"], label="Validation AP")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc_pr_calibration(y_true: np.ndarray, probs: np.ndarray, output_dir: Path, prefix: str) -> None:
    """Generate ROC, precision-recall, and calibration plots for Section 5.1."""
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=300)
        plt.close()

        precision, recall, _ = precision_recall_curve(y_true, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_pr_curve.png", dpi=300)
        plt.close()

    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Calibration curve")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_calibration_curve.png", dpi=300)
    plt.close()


def plot_confusion(y_true: np.ndarray, probs: np.ndarray, threshold: float, path: Path) -> None:
    """Plot a confusion matrix for the chosen operating threshold."""
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_metric_bars(metrics_df: pd.DataFrame, path: Path) -> None:
    """Compare ablation metrics in a compact figure for Section 5.1."""
    metrics = ["average_precision", "roc_auc", "f1", "balanced_accuracy"]
    x = np.arange(len(metrics_df))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        values = metrics_df[metric].fillna(0.0).values
        plt.bar(x + i * width, values, width=width, label=metric)

    plt.xticks(x + width * (len(metrics) - 1) / 2, metrics_df["run_name"].tolist(), rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Metric value")
    plt.title("Model comparison and ablation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_spatial_surface(
    data: DataBundle,
    values: np.ndarray,
    path: Path,
    title: str,
    annotate_top_k: int = 10,
) -> None:
    """
    Draw a simple georeferenced scatter surface suitable for a manuscript figure
    when polygon layers are not provided.
    """
    x = data.coords[:, 0]
    y = data.coords[:, 1]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=values, s=40 + 60 * normalize_minmax(values))
    plt.colorbar(sc, label="Value")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(title)

    top_idx = np.argsort(-values)[:annotate_top_k]
    for idx in top_idx:
        plt.annotate(data.node_ids[idx], (x[idx], y[idx]), fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_persistence_image_panel(topology: TopologyBundle, output_dir: Path, n_show: int = 6) -> None:
    """
    Save representative H1 persistence images for qualitative topological interpretation.
    """
    reps = topology.representative_nodes[:n_show]
    ncols = min(3, len(reps))
    nrows = int(math.ceil(len(reps) / ncols))
    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, node_id in enumerate(reps, start=1):
        plt.subplot(nrows, ncols, i)
        plt.imshow(topology.persistence_images[node_id][1], aspect="auto", origin="lower")
        plt.title(f"{node_id} | H1 PI")
        plt.xlabel("Birth axis")
        plt.ylabel("Persistence axis")
    plt.tight_layout()
    plt.savefig(output_dir / "representative_h1_persistence_images.png", dpi=300)
    plt.close()


def plot_betti_curves_panel(topology: TopologyBundle, output_dir: Path, n_show: int = 6) -> None:
    """
    Save representative Betti curves for H0 and H1.
    """
    reps = topology.representative_nodes[:n_show]
    ncols = min(3, len(reps))
    nrows = int(math.ceil(len(reps) / ncols))
    plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, node_id in enumerate(reps, start=1):
        plt.subplot(nrows, ncols, i)
        plt.plot(topology.betti_curves[node_id][0], label="Betti 0")
        plt.plot(topology.betti_curves[node_id][1], label="Betti 1")
        plt.title(node_id)
        plt.xlabel("Filtration grid index")
        plt.ylabel("Betti value")
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "representative_betti_curves.png", dpi=300)
    plt.close()


def export_mapper_graph(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    cfg: Dict[str, Any],
) -> None:
    """
    Generate a KeplerMapper HTML summary of the latent embedding space.
    This artifact is useful for Section 5.2 because it reveals whether latent
    neighborhoods of risk are structurally segmented or bridged.
    """
    if km is None:
        write_text(
            output_dir / "mapper_graph_not_generated.txt",
            "kmapper is not installed. Install the package listed in requirements.txt to generate the Mapper HTML artifact.",
        )
        return

    mapper_cfg = cfg["topology"].get("mapper", {})
    if not mapper_cfg.get("enabled", True):
        return

    mapper = km.KeplerMapper(verbose=0)
    lens = embeddings[:, : min(2, embeddings.shape[1])]
    graph = mapper.map(
        lens=lens,
        X=embeddings,
        cover=km.Cover(n_cubes=int(mapper_cfg.get("n_cubes", 10)), perc_overlap=float(mapper_cfg.get("overlap_perc", 0.35))),
        clusterer=KMeans(n_clusters=2, random_state=cfg["project"]["seed"], n_init=10),
    )
    html_path = output_dir / "mapper_latent_risk_graph.html"
    mapper.visualize(
        graph,
        path_html=str(html_path),
        title="Mapper summary of latent topology-aware embeddings",
        color_values=labels,
        color_function_name="Risk label",
    )


def plot_pathway_graph(pathway_graph: nx.DiGraph, data: DataBundle, path: Path) -> None:
    """Plot the subgraph induced by the selected high-scoring risk pathways."""
    plt.figure(figsize=(9, 7))
    pos = {row["node_id"]: (row["x"], row["y"]) for _, row in data.nodes_df[["node_id", "x", "y"]].iterrows()}
    if pathway_graph.number_of_nodes() == 0:
        write_text(path.with_suffix(".txt"), "No risk pathways were extracted under the configured settings.")
        plt.close()
        return
    nx.draw_networkx(pathway_graph, pos=pos, with_labels=True, node_size=240, font_size=7, arrows=True)
    plt.title("High-priority risk pathways")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_priority_rankings(priority_df: pd.DataFrame, path: Path, top_k: int = 20) -> None:
    """Plot top intervention priorities as a simple ranked bar chart."""
    sub = priority_df.head(top_k).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(sub)), sub["priority_score"].values)
    plt.xticks(np.arange(len(sub)), sub["node_id"].tolist(), rotation=60)
    plt.ylabel("Priority score")
    plt.title("Top intervention priorities")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_scenario_comparison(scenario_df: pd.DataFrame, path: Path) -> None:
    """Plot scenario-wise changes in mean risk and mean priority."""
    plt.figure(figsize=(8, 5))
    x = np.arange(len(scenario_df))
    width = 0.35
    plt.bar(x - width / 2, scenario_df["mean_risk_probability"].values, width=width, label="Mean risk")
    plt.bar(x + width / 2, scenario_df["mean_priority_score"].values, width=width, label="Mean priority")
    plt.xticks(x, scenario_df["scenario_name"].tolist(), rotation=20)
    plt.ylabel("Value")
    plt.title("Scenario comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# --------------------------------------------------------------------------------------
# End-to-end pipeline orchestration
# --------------------------------------------------------------------------------------

def run_pipeline(cfg: Dict[str, Any]) -> None:
    """
    Execute the complete pipeline and export all manuscript-ready artifacts.

    Folder logic:
        outputs/5.1 -> predictive performance
        outputs/5.2 -> interpretation of topological signatures
        outputs/5.3 -> intervention prioritization and scenarios
    """
    output_root = Path(cfg["project"].get("output_dir", "outputs"))
    out_51 = output_root / "5.1"
    out_52 = output_root / "5.2"
    out_53 = output_root / "5.3"
    for path in [output_root, out_51, out_52, out_53]:
        safe_mkdir(path)

    setup_logger(output_root)
    set_seed(int(cfg["project"].get("seed", 42)))

    logging.info("Starting explainable topological deep learning pipeline.")

    data = load_and_preprocess_data(cfg)
    graph = build_graph_bundle(data, cfg)
    topology = compute_topological_embeddings(data, graph, cfg)

    train_idx, val_idx, test_idx = create_spatial_split(data, cfg)
    full_art = train_model(data, graph, topology.topo_matrix, cfg, train_idx, val_idx, test_idx, run_name="full_topology_model")

    # ----------------------------- Section 5.1 outputs -----------------------------
    metrics = compute_metrics(data.labels[test_idx], full_art.test_probabilities, full_art.threshold)
    metrics_df = pd.DataFrame([metrics])
    save_dataframe(metrics_df, out_51 / "predictive_metrics.csv")
    save_dataframe(full_art.history, out_51 / "training_history.csv")
    plot_training_history(full_art.history, out_51 / "training_history.png", "Training dynamics of the full topology-aware model")
    plot_roc_pr_calibration(data.labels[test_idx], full_art.test_probabilities, out_51, "full_model_test")
    plot_confusion(data.labels[test_idx], full_art.test_probabilities, full_art.threshold, out_51 / "confusion_matrix.png")

    ablation_df, ablation_probs = run_ablation_suite(data, graph, topology, cfg, train_idx, val_idx, test_idx)
    save_dataframe(ablation_df, out_51 / "ablation_metrics.csv")
    plot_metric_bars(ablation_df, out_51 / "ablation_metrics.png")

    # Export node-level test predictions to support later manuscript tables.
    test_rows = []
    for local_idx, global_idx in enumerate(test_idx):
        test_rows.append(
            {
                "node_id": data.node_ids[global_idx],
                "source_type": data.nodes_df.iloc[global_idx]["source_type"],
                "district": data.nodes_df.iloc[global_idx]["district"] if "district" in data.nodes_df.columns else "NA",
                "true_label": int(data.labels[global_idx]),
                "predicted_probability": float(full_art.test_probabilities[local_idx]),
                "predicted_label": int(full_art.test_predictions[local_idx]),
            }
        )
    save_dataframe(pd.DataFrame(test_rows), out_51 / "test_predictions.csv")

    # ----------------------------- Section 5.2 outputs -----------------------------
    device = default_device()
    X = torch.tensor(data.feature_matrix, dtype=torch.float32, device=device)
    T = torch.tensor(topology.topo_matrix, dtype=torch.float32, device=device)
    S = torch.tensor(data.source_type_ids.astype(np.int64), dtype=torch.long, device=device)
    A_rel = {rel: torch.tensor(adj, dtype=torch.float32, device=device) for rel, adj in graph.adjacency_by_relation.items()}

    representative_idx = [data.node_ids.index(nid) for nid in topology.representative_nodes[: min(12, len(topology.representative_nodes))]]
    feature_attr = integrated_gradients(full_art.model, X, T, S, A_rel, representative_idx, steps=32)
    topo_attr = topological_attribution(full_art.model, X, T, S, A_rel)

    explanation_df = build_local_explanation_table(
        data=data,
        topology_attr=topo_attr,
        feature_attr=feature_attr,
        probabilities=full_art.full_probabilities,
        top_k=15,
    )
    save_dataframe(explanation_df, out_52 / "local_explanations_top_nodes.csv")

    plot_spatial_surface(data, full_art.full_probabilities, out_52 / "predicted_risk_surface.png", "Predicted pollution-risk surface")
    plot_spatial_surface(data, topo_attr, out_52 / "topological_criticality_surface.png", "Topological criticality surface")
    plot_persistence_image_panel(topology, out_52, n_show=6)
    plot_betti_curves_panel(topology, out_52, n_show=6)
    export_mapper_graph(full_art.latent_embeddings, data.labels, out_52, cfg)

    # Additional comparative plot: risk vs topological attribution.
    plt.figure(figsize=(7, 5))
    plt.scatter(full_art.full_probabilities, topo_attr, s=35)
    plt.xlabel("Predicted risk probability")
    plt.ylabel("Topological attribution")
    plt.title("Risk-topology relationship")
    plt.tight_layout()
    plt.savefig(out_52 / "risk_vs_topological_attribution.png", dpi=300)
    plt.close()

    pathways_df, pathway_graph = extract_risk_pathways(data, graph, full_art.full_probabilities, topo_attr, cfg)
    save_dataframe(pathways_df, out_52 / "risk_pathways.csv")
    plot_pathway_graph(pathway_graph, data, out_52 / "risk_pathways_graph.png")

    # ----------------------------- Section 5.3 outputs -----------------------------
    priority_df = compute_intervention_priority_table(data, full_art.full_probabilities, topo_attr)
    save_dataframe(priority_df, out_53 / "intervention_priority_table.csv")
    plot_spatial_surface(data, priority_df["priority_score"].values, out_53 / "intervention_priority_surface.png", "Intervention-priority surface")
    plot_priority_rankings(priority_df, out_53 / "top_intervention_priorities.png", top_k=20)

    # Scenario analysis.
    scenario_rows = []
    for scenario_name, scenario_cfg in cfg.get("scenarios", {"baseline": {}}).items():
        scenario_data, scenario_graph = apply_scenario_to_data_and_graph(data, graph, scenario_cfg)

        # Recompute topology because the scenario may change node attributes or edge structure.
        scenario_topology = compute_topological_embeddings(scenario_data, scenario_graph, cfg)

        X_sc = torch.tensor(scenario_data.feature_matrix, dtype=torch.float32, device=device)
        T_sc = torch.tensor(scenario_topology.topo_matrix, dtype=torch.float32, device=device)
        S_sc = torch.tensor(scenario_data.source_type_ids.astype(np.int64), dtype=torch.long, device=device)
        A_rel_sc = {rel: torch.tensor(adj, dtype=torch.float32, device=device) for rel, adj in scenario_graph.adjacency_by_relation.items()}

        with torch.no_grad():
            probs_sc = torch.sigmoid(full_art.model(X_sc, T_sc, S_sc, A_rel_sc)).detach().cpu().numpy()

        topo_attr_sc = topological_attribution(full_art.model, X_sc, T_sc, S_sc, A_rel_sc)
        priority_sc = compute_intervention_priority_table(scenario_data, probs_sc, topo_attr_sc)

        scenario_rows.append(
            {
                "scenario_name": scenario_name,
                "mean_risk_probability": float(np.mean(probs_sc)),
                "mean_topological_attribution": float(np.mean(topo_attr_sc)),
                "mean_priority_score": float(np.mean(priority_sc["priority_score"].values)),
                "top_priority_node": str(priority_sc.iloc[0]["node_id"]) if len(priority_sc) > 0 else "NA",
            }
        )

        # Save the top nodes for each scenario as individual CSV files.
        save_dataframe(priority_sc.head(25), out_53 / f"scenario_{scenario_name}_top25_priorities.csv")

    scenario_df = pd.DataFrame(scenario_rows).sort_values("mean_priority_score", ascending=False)
    save_dataframe(scenario_df, out_53 / "scenario_summary.csv")
    plot_scenario_comparison(scenario_df, out_53 / "scenario_comparison.png")

    # ----------------------------- Audit summaries -----------------------------
    summary_text = textwrap.dedent(
        f"""
        Explainable Topological Deep Learning Pipeline Summary
        =====================================================

        Nodes processed: {len(data.node_ids)}
        Feature count: {len(data.feature_names)}
        Topological feature count: {topology.topo_matrix.shape[1]}
        Relation types: {", ".join(graph.relation_names)}

        Output folders:
            - {out_51}
            - {out_52}
            - {out_53}

        Main evaluation metrics on the held-out test split:
            {json.dumps(metrics, indent=4)}

        This audit file is intended to support transparent manuscript writing.
        """
    ).strip()
    write_text(output_root / "run_summary.txt", summary_text)

    logging.info("Pipeline finished successfully. Outputs saved under %s", output_root)


# --------------------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Explainable topological deep learning for pollution risk mapping in multi-source urban water systems."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_example.yaml",
        help="Path to YAML or JSON configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(Path(args.config))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
