#!/usr/bin/env python3
"""
Betti-Curve Analysis of Wastewater Reuse Scenarios for Sustainable Urban Water Management
========================================================================================

This script implements a fully reproducible computational pipeline for comparing treated
wastewater reuse scenarios through exact Betti-curve summaries, perturbation-based
robustness analysis, and environmentally constrained compromise ranking.

Design philosophy
-----------------
The script was written to support an article in which scenario comparison must go beyond
average indicators such as mean cost or mean recovered volume. Instead of relying only
on averages, each scenario is represented as a *cloud of plausible operating states* in
a multivariate sustainability feature space. A graph filtration is built from pairwise
distances among states, the clique complex of the threshold graph is computed exactly,
and Betti curves are extracted across the filtration.

The analysis is intentionally self-contained:
    * It does not depend on GUDHI.
    * It does not require any specialized topological package.
    * It computes Betti-0 and Betti-1 directly with standard scientific Python tools.
    * It exports figures and tables already organized in folders 5.1, 5.2, and 5.3,
      so they can be inserted directly into the corresponding subsections of a paper.

Main outputs
------------
results/5.1
    - Figure_1_beta0_curves.png
    - Figure_2_beta1_curves.png
    - Figure_3_state_cloud_projection.png
    - Table_1_topological_summary.csv
    - Table_1_topological_summary.xlsx

results/5.2
    - Figure_4_sensitivity_heatmap.png
    - Figure_5_stability_vs_sensitivity.png
    - Table_2_robustness_sensitivity.csv
    - Table_2_robustness_sensitivity.xlsx

results/5.3
    - Figure_6_tradeoff_space.png
    - Figure_7_composite_ranking.png
    - Table_3_composite_ranking.csv
    - Table_3_composite_ranking.xlsx

Additional exports
------------------
    - simulated_states.csv
    - curves/S1_betti_curve.csv ... curves/S6_betti_curve.csv
    - run_summary.txt

Mathematical interpretation
---------------------------
For each scenario s and state vectors x_i^(s), the script computes:

1. Standardization
       z_ij = (x_ij - mu_j) / sigma_j

2. Weighted Euclidean distance
       d_ik^(s) = sqrt( sum_j w_j * (z_ij^(s) - z_kj^(s))^2 )

3. Threshold graph at filtration value epsilon
       G_s(epsilon) = (V_s, E_s(epsilon))
       E_s(epsilon) = {(i,k) : d_ik^(s) <= epsilon}

4. Clique complex of the threshold graph
       K_s(epsilon) = Cl(G_s(epsilon))

5. Betti curves
       beta_p^(s)(epsilon) = rank H_p(K_s(epsilon); Z_2),  p in {0,1}

Because the complex is the clique complex of a graph, Betti-1 can be computed exactly as

       beta_1 = |E| - |V| + beta_0 - rank_Z2(partial_2)

where partial_2 is the triangle boundary matrix over Z_2.

The code therefore remains transparent and auditable: every reported scalar summary can
be traced back to explicit graph and clique-complex constructions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ======================================================================================
# 1. Global configuration
# ======================================================================================

FEATURES: List[str] = [
    "Availability",
    "TreatmentCost",
    "EnergyIntensity",
    "OverloadRisk",
    "ComplianceReliability",
    "DemandCoverage",
]

# Bounds are intentionally normalized or semi-normalized to [0, 1]-like ranges.
# This keeps the simulation interpretable and prevents unrealistic samples from
# dominating the topology.
BOUNDS: Dict[str, Tuple[float, float]] = {
    "Availability": (0.20, 0.95),
    "TreatmentCost": (0.20, 1.00),
    "EnergyIntensity": (0.20, 0.95),
    "OverloadRisk": (0.05, 0.95),
    "ComplianceReliability": (0.65, 0.995),
    "DemandCoverage": (0.20, 0.95),
}

# Feature weights make the distance metric policy-sensitive. Higher weights place
# stronger emphasis on variables that are especially relevant to environmental
# reliability and scenario differentiation.
FEATURE_WEIGHTS = np.array([1.15, 1.00, 0.90, 1.20, 1.10, 1.05], dtype=float)

# A dense filtration grid allows smooth Betti curves without making the analysis
# computationally expensive.
THRESHOLDS = np.linspace(0.25, 2.80, 28)

# The perturbation plan focuses on the three variables emphasized by the article:
# availability, treatment cost, and overload risk.
PERTURBATIONS: List[Tuple[str, float]] = [
    ("Availability", -0.10),
    ("Availability", +0.10),
    ("TreatmentCost", -0.10),
    ("TreatmentCost", +0.10),
    ("OverloadRisk", -0.10),
    ("OverloadRisk", +0.10),
]

# Correlation structure shared by all scenarios. This is not meant to be universal;
# it is a synthetic but policy-plausible correlation template:
#   - higher cost tends to co-move with higher energy intensity,
#   - availability tends to support demand coverage,
#   - overload risk tends to worsen when compliance weakens, etc.
CORR = np.array([
    [ 1.00, -0.35, -0.25, -0.30,  0.30,  0.55],
    [-0.35,  1.00,  0.60,  0.25, -0.20, -0.25],
    [-0.25,  0.60,  1.00,  0.20, -0.15, -0.20],
    [-0.30,  0.25,  0.20,  1.00, -0.45, -0.25],
    [ 0.30, -0.20, -0.15, -0.45,  1.00,  0.35],
    [ 0.55, -0.25, -0.20, -0.25,  0.35,  1.00],
], dtype=float)

SCENARIOS: Dict[str, Dict[str, object]] = {
    "S1": {
        "name": "Conventional discharge-oriented reuse",
        "mean": [0.40, 0.46, 0.41, 0.67, 0.81, 0.33],
        "std":  [0.14, 0.11, 0.10, 0.13, 0.08, 0.15],
    },
    "S2": {
        "name": "Urban non-potable reuse",
        "mean": [0.63, 0.58, 0.55, 0.38, 0.91, 0.59],
        "std":  [0.11, 0.10, 0.09, 0.10, 0.05, 0.10],
    },
    "S3": {
        "name": "Industrial symbiosis reuse",
        "mean": [0.69, 0.52, 0.49, 0.34, 0.94, 0.64],
        "std":  [0.10, 0.09, 0.08, 0.09, 0.04, 0.09],
    },
    "S4": {
        "name": "Aquifer recharge buffer",
        "mean": [0.57, 0.64, 0.60, 0.29, 0.96, 0.51],
        "std":  [0.09, 0.12, 0.11, 0.07, 0.03, 0.09],
    },
    "S5": {
        "name": "Peri-urban irrigation reuse",
        "mean": [0.74, 0.36, 0.34, 0.61, 0.84, 0.72],
        "std":  [0.15, 0.08, 0.07, 0.15, 0.07, 0.12],
    },
    "S6": {
        "name": "Hybrid adaptive reuse",
        "mean": [0.81, 0.57, 0.50, 0.24, 0.97, 0.79],
        "std":  [0.08, 0.09, 0.08, 0.06, 0.03, 0.07],
    },
}


# ======================================================================================
# 2. Utility functions
# ======================================================================================

def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compatibility wrapper for numerical integration.

    Newer NumPy versions expose np.trapezoid while older versions expose np.trapz.
    The two are equivalent for the purposes of this workflow.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def clip_columns(X: np.ndarray) -> np.ndarray:
    """
    Clip each feature column to its admissible interval.

    Clipping is important because multivariate normal simulation can generate values
    outside the intended policy range. Without clipping, the distance geometry could
    become dominated by unrealistic tail realizations.
    """
    X = np.asarray(X, dtype=float).copy()
    for j, feat in enumerate(FEATURES):
        lo, hi = BOUNDS[feat]
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X


def scenario_dataframe(code: str, X: np.ndarray) -> pd.DataFrame:
    """
    Convert one scenario matrix into a labeled DataFrame.

    Parameters
    ----------
    code:
        Scenario code such as 'S1' or 'S6'.
    X:
        Array of shape (n_samples, n_features).

    Returns
    -------
    pandas.DataFrame
        Tidy table with scenario metadata and the six core features.
    """
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "Scenario", code)
    df.insert(1, "ScenarioName", str(SCENARIOS[code]["name"]))
    return df


def minmax_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Normalize a metric into the [0, 1] interval.

    The function supports both benefit-type and cost-type indicators. If all values are
    identical, the metric is treated as non-discriminatory and every scenario receives 1.
    """
    s = series.astype(float)
    if float(s.max()) == float(s.min()):
        norm = pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    else:
        norm = (s - s.min()) / (s.max() - s.min())
    if not higher_is_better:
        norm = 1.0 - norm
    return norm


def first_eps_where(condition_series: pd.Series, eps_series: pd.Series) -> float:
    """
    Return the first filtration threshold for which a logical condition is satisfied.

    If the condition is never satisfied, NaN is returned. This is useful for summaries
    such as the first epsilon at which the scenario becomes globally connected.
    """
    idx = np.where(condition_series.to_numpy())[0]
    return float(eps_series.iloc[idx[0]]) if len(idx) else float("nan")


def relative_change(new: float, base: float, eps: float = 1e-9) -> float:
    """
    Absolute relative change with small-denominator protection.
    """
    return abs(float(new) - float(base)) / (abs(float(base)) + eps)


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    """
    Create the directory structure required by the manuscript.

    Returns a dictionary with the most important directory paths so that downstream
    code does not need to reconstruct them repeatedly.
    """
    results_dir = base_dir / "results"
    dir_51 = results_dir / "5.1"
    dir_52 = results_dir / "5.2"
    dir_53 = results_dir / "5.3"
    curves_dir = base_dir / "curves"

    for folder in [base_dir, results_dir, dir_51, dir_52, dir_53, curves_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": base_dir,
        "results_dir": results_dir,
        "dir_51": dir_51,
        "dir_52": dir_52,
        "dir_53": dir_53,
        "curves_dir": curves_dir,
    }


def write_dataframe(df: pd.DataFrame, path_csv: Path, path_xlsx: Path | None = None) -> None:
    """
    Export a DataFrame to CSV and optionally to XLSX.

    CSV is convenient for transparent inspection and version control.
    XLSX is convenient for authors who will later insert the table into a paper.
    """
    df.to_csv(path_csv, index=False)
    if path_xlsx is not None:
        with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)


# ======================================================================================
# 3. Scenario simulation
# ======================================================================================

def simulate_scenarios(n_samples: int = 60, seed: int = 123) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Simulate uncertain operating states for every wastewater reuse scenario.

    The simulation is multivariate normal with scenario-specific means and standard
    deviations, but with a shared correlation matrix. Each scenario is therefore
    allowed to occupy a distinct region of the policy space while still respecting
    a common relational logic among features.

    Returns
    -------
    scenario_data:
        Dictionary mapping scenario code -> array of simulated states.
    combined_df:
        Single stacked DataFrame for all simulated states.
    """
    rng = np.random.default_rng(seed)
    scenario_data: Dict[str, np.ndarray] = {}
    all_frames: List[pd.DataFrame] = []

    for code, cfg in SCENARIOS.items():
        mean = np.asarray(cfg["mean"], dtype=float)
        std = np.asarray(cfg["std"], dtype=float)
        cov = np.outer(std, std) * CORR
        X = rng.multivariate_normal(mean, cov, size=n_samples)
        X = clip_columns(X)
        scenario_data[code] = X
        all_frames.append(scenario_dataframe(code, X))

    combined_df = pd.concat(all_frames, ignore_index=True)
    return scenario_data, combined_df


def standardize_scenarios(scenario_data: Dict[str, np.ndarray]) -> Tuple[StandardScaler, Dict[str, np.ndarray]]:
    """
    Standardize all scenarios using one global scaler.

    Global standardization is essential because the article compares scenarios in a
    common feature space. Standardizing each scenario separately would erase some of
    the cross-scenario differences that the ranking is supposed to interpret.
    """
    allX = np.vstack([scenario_data[k] for k in SCENARIOS.keys()])
    scaler = StandardScaler().fit(allX)
    standardized = {k: scaler.transform(v) for k, v in scenario_data.items()}
    return scaler, standardized


# ======================================================================================
# 4. Topological core: exact Betti-0 and Betti-1 on clique complexes
# ======================================================================================

def weighted_distance_matrix(Z: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Compute a weighted Euclidean distance matrix.

    If D is the distance matrix, then the threshold graph at epsilon contains edge
    (i, k) whenever D[i, k] <= epsilon.
    """
    if weights is None:
        weights = np.ones(Z.shape[1], dtype=float)
    W = np.sqrt(np.asarray(weights, dtype=float))
    Zw = Z * W
    return squareform(pdist(Zw, metric="euclidean"))


def gf2_rank_bitmasks(bitmasks: Iterable[int]) -> int:
    """
    Compute rank over Z_2 using integer bitmasks.

    Why bitmasks?
    -------------
    The boundary of each triangle in a clique complex can be written as the XOR of
    its three incident edges. Instead of building a dense binary matrix, the script
    stores each boundary row as an integer whose bits indicate which edges are present.
    Gaussian elimination over Z_2 then becomes repeated XOR reduction.

    This approach is compact, exact, and very fast for the small-to-medium complexes
    generated by the scenario clouds used in this study.
    """
    basis: Dict[int, int] = {}
    rank = 0
    for x in bitmasks:
        x = int(x)
        while x:
            pivot = x.bit_length() - 1
            pivot_row = basis.get(pivot)
            if pivot_row is None:
                basis[pivot] = x
                rank += 1
                break
            x ^= pivot_row
    return rank


def betti_graph_clique_exact(D: np.ndarray, eps: float) -> Dict[str, float]:
    """
    Compute beta_0 and beta_1 of the clique complex of the threshold graph D <= eps.

    Coefficients are in Z_2.

    The method proceeds in four explicit steps:
        1. Build the threshold graph.
        2. Compute connected components to obtain beta_0.
        3. Enumerate all triangles, which are the 2-simplices of the clique complex.
        4. Use the Euler-type identity

               beta_1 = |E| - |V| + beta_0 - rank_Z2(partial_2)

           to obtain beta_1 exactly.

    Returns
    -------
    dict
        beta0, beta1, n_vertices, n_edges, n_triangles, rank_b2
    """
    n = int(D.shape[0])

    # ------------------------------------------------------------------
    # Step 1. Build the threshold graph at the current epsilon.
    # ------------------------------------------------------------------
    edges: List[Tuple[int, int]] = []
    neighbors: List[set[int]] = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                edges.append((i, j))
                neighbors[i].add(j)
                neighbors[j].add(i)

    # ------------------------------------------------------------------
    # Step 2. Connected components for beta_0.
    # ------------------------------------------------------------------
    if not edges:
        beta0 = n
    else:
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for u, v in edges:
            union(u, v)

        beta0 = len({find(i) for i in range(n)})

    # ------------------------------------------------------------------
    # Step 3. Triangle enumeration. Every 3-clique in the graph is a
    #         filled 2-simplex in the clique complex.
    # ------------------------------------------------------------------
    m = len(edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    triangle_boundary_bitmasks: List[int] = []
    triangles = 0

    for u in range(n):
        candidate_neighbors = sorted(v for v in neighbors[u] if v > u)
        for idx, v in enumerate(candidate_neighbors):
            # The intersection identifies w such that (u,v,w) is a triangle.
            common = [w for w in candidate_neighbors[idx + 1:] if w in neighbors[v]]
            for w in common:
                triangles += 1
                e1 = edge_index[(u, v)]
                e2 = edge_index[(u, w)]
                e3 = edge_index[(v, w)]
                mask = (1 << e1) ^ (1 << e2) ^ (1 << e3)
                triangle_boundary_bitmasks.append(mask)

    # ------------------------------------------------------------------
    # Step 4. Rank of the triangle boundary map over Z_2, then beta_1.
    # ------------------------------------------------------------------
    rank_b2 = gf2_rank_bitmasks(triangle_boundary_bitmasks)
    beta1 = m - n + beta0 - rank_b2

    return {
        "beta0": int(beta0),
        "beta1": int(beta1),
        "n_vertices": int(n),
        "n_edges": int(m),
        "n_triangles": int(triangles),
        "rank_b2": int(rank_b2),
    }


def betti_curve(
    Z: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
    weights: np.ndarray = FEATURE_WEIGHTS,
) -> pd.DataFrame:
    """
    Compute the full Betti curve for one standardized scenario cloud.

    The result is a table with one row per filtration threshold.
    """
    D = weighted_distance_matrix(Z, weights)
    rows: List[Dict[str, float]] = []
    for eps in thresholds:
        stats = betti_graph_clique_exact(D, float(eps))
        rows.append({"eps": float(eps), **stats})
    return pd.DataFrame(rows)


def summarize_curve(curve_df: pd.DataFrame) -> Dict[str, float]:
    """
    Derive compact scalar summaries from a Betti curve.

    The summary intentionally mixes cumulative and event-based descriptors:
        - AUBC0 / AUBC1 measure cumulative fragmentation / cycle intensity.
        - eps_conn is the first threshold with one connected component.
        - eps_loop is the first threshold at which a loop appears.
        - peak_beta1 measures the maximum size of the loop regime.
        - volatility metrics summarize the rate of topological change.
    """
    eps = curve_df["eps"].to_numpy(dtype=float)
    beta0 = curve_df["beta0"].to_numpy(dtype=float)
    beta1 = curve_df["beta1"].to_numpy(dtype=float)

    d_beta0 = np.diff(beta0)
    d_beta1 = np.diff(beta1)

    return {
        "AUBC0": trapz(beta0, eps),
        "AUBC1": trapz(beta1, eps),
        "eps50": first_eps_where(curve_df["beta0"] <= beta0[0] / 2.0, curve_df["eps"]),
        "eps_conn": first_eps_where(curve_df["beta0"] == 1, curve_df["eps"]),
        "eps_loop": first_eps_where(curve_df["beta1"] > 0, curve_df["eps"]),
        "peak_beta1": int(beta1.max()),
        "peak_eps": float(curve_df.loc[curve_df["beta1"].idxmax(), "eps"]),
        "beta0_volatility": float(np.mean(np.abs(d_beta0))),
        "beta1_volatility": float(np.mean(np.abs(d_beta1))),
    }


def baseline_environmental_metrics(raw_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute average environmental and service metrics for one scenario.
    """
    avg = raw_df[FEATURES].mean()
    return {
        "mean_availability": float(avg["Availability"]),
        "mean_treatment_cost": float(avg["TreatmentCost"]),
        "mean_energy_intensity": float(avg["EnergyIntensity"]),
        "mean_overload_risk": float(avg["OverloadRisk"]),
        "mean_compliance_reliability": float(avg["ComplianceReliability"]),
        "mean_demand_coverage": float(avg["DemandCoverage"]),
    }


# ======================================================================================
# 5. Sensitivity analysis
# ======================================================================================

def apply_feature_perturbation(X: np.ndarray, feature: str, frac: float) -> np.ndarray:
    """
    Apply a multiplicative perturbation to one feature of one scenario.

    Example
    -------
    frac = -0.10 means a 10% decrease.
    frac = +0.10 means a 10% increase.
    """
    Xp = np.asarray(X, dtype=float).copy()
    j = FEATURES.index(feature)
    Xp[:, j] = Xp[:, j] * (1.0 + frac)
    lo, hi = BOUNDS[feature]
    Xp[:, j] = np.clip(Xp[:, j], lo, hi)
    return Xp


def sensitivity_analysis(
    scenario_data: Dict[str, np.ndarray],
    thresholds: np.ndarray = THRESHOLDS,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """
    Stress-test each scenario by perturbing one feature at a time.

    Important design choice
    -----------------------
    After perturbing one scenario, the code re-standardizes the *entire* dataset. This
    preserves the article's interpretation that scenarios live in a shared comparison
    space. If only the perturbed scenario were re-standardized independently, the
    perturbation effect would be partially hidden.

    Returns
    -------
    sens_detail:
        One row per (scenario, feature, perturbation sign).
    sens_agg:
        Mean sensitivity summaries aggregated by scenario.
    baseline_curves:
        Baseline Betti curves for all scenarios.
    baseline_summaries:
        Compact baseline summaries for all scenarios.
    """
    _, base_Z = standardize_scenarios(scenario_data)
    baseline_curves = {k: betti_curve(base_Z[k], thresholds) for k in SCENARIOS}
    baseline_summaries = {k: summarize_curve(v) for k, v in baseline_curves.items()}

    rows: List[Dict[str, object]] = []

    for code, X in scenario_data.items():
        base_sum = baseline_summaries[code]

        for feature, frac in PERTURBATIONS:
            perturbed_data = {kk: vv.copy() for kk, vv in scenario_data.items()}
            perturbed_data[code] = apply_feature_perturbation(X, feature, frac)

            _, Z = standardize_scenarios(perturbed_data)
            curve_df = betti_curve(Z[code], thresholds)
            curve_sum = summarize_curve(curve_df)

            row = {
                "Scenario": code,
                "ScenarioName": str(SCENARIOS[code]["name"]),
                "Feature": feature,
                "Change": float(frac),
                "AUBC0_rel_change": relative_change(curve_sum["AUBC0"], base_sum["AUBC0"]),
                "AUBC1_rel_change": relative_change(curve_sum["AUBC1"], base_sum["AUBC1"]),
                "eps_conn_rel_change": relative_change(curve_sum["eps_conn"], base_sum["eps_conn"]),
                "peak_beta1_rel_change": relative_change(curve_sum["peak_beta1"], base_sum["peak_beta1"]),
            }

            row["AggregateSensitivity"] = float(np.mean([
                row["AUBC0_rel_change"],
                row["AUBC1_rel_change"],
                row["eps_conn_rel_change"],
                row["peak_beta1_rel_change"],
            ]))
            rows.append(row)

    sens_detail = pd.DataFrame(rows)

    sens_agg = (
        sens_detail
        .groupby(["Scenario", "ScenarioName"], as_index=False)
        .agg({
            "AggregateSensitivity": "mean",
            "AUBC0_rel_change": "mean",
            "AUBC1_rel_change": "mean",
            "eps_conn_rel_change": "mean",
            "peak_beta1_rel_change": "mean",
        })
        .rename(columns={
            "AggregateSensitivity": "MeanSensitivity",
            "AUBC0_rel_change": "MeanAUBC0Sensitivity",
            "AUBC1_rel_change": "MeanAUBC1Sensitivity",
            "eps_conn_rel_change": "MeanConnectionSensitivity",
            "peak_beta1_rel_change": "MeanLoopPeakSensitivity",
        })
    )

    return sens_detail, sens_agg, baseline_curves, baseline_summaries


# ======================================================================================
# 6. Composite indices and ranking
# ======================================================================================

def compute_indices(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw topological and environmental summaries into normalized indices.

    Interpretation of the two most important aggregate scores
    ---------------------------------------------------------
    TopologicalStabilityScore:
        Rewards compactness, early connectivity, and parsimonious loop structure.

    EnvironmentalSafetyScore:
        Rewards low overload risk, high compliance reliability, lower energy
        intensity, and stronger availability.

    The weights were selected to reflect the article's planning logic rather than to
    claim universal optimality. They can be modified easily for other cities or policy
    priorities.
    """
    out = summary_df.copy()

    out["TopoCompactness"] = (
        minmax_normalize(out["AUBC0"], higher_is_better=False) * 0.45 +
        minmax_normalize(out["eps_conn"], higher_is_better=False) * 0.35 +
        minmax_normalize(out["AUBC1"], higher_is_better=False) * 0.20
    )

    out["LoopParsimony"] = minmax_normalize(out["peak_beta1"], higher_is_better=False)

    out["TopologicalStabilityScore"] = (
        0.75 * out["TopoCompactness"] +
        0.25 * out["LoopParsimony"]
    )

    out["EnvironmentalSafetyScore"] = (
        minmax_normalize(out["mean_overload_risk"], higher_is_better=False) * 0.40 +
        minmax_normalize(out["mean_compliance_reliability"], higher_is_better=True) * 0.30 +
        minmax_normalize(out["mean_energy_intensity"], higher_is_better=False) * 0.15 +
        minmax_normalize(out["mean_availability"], higher_is_better=True) * 0.15
    )

    out["ServiceCoverageScore"] = (
        minmax_normalize(out["mean_demand_coverage"], higher_is_better=True) * 0.60 +
        minmax_normalize(out["mean_availability"], higher_is_better=True) * 0.40
    )

    out["EconomicScore"] = minmax_normalize(out["mean_treatment_cost"], higher_is_better=False)

    return out


def compute_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the final environmentally constrained ranking.

    Admissibility logic
    -------------------
    A scenario is marked feasible only if it simultaneously satisfies:
        - overload risk <= 0.55
        - treatment cost <= 0.60
        - compliance reliability >= 0.88

    Even infeasible scenarios are retained in the ranking because the article is not
    only interested in binary accept/reject decisions. It is also interested in how
    attractive but environmentally strained scenarios compare to robust alternatives.

    Composite score
    ---------------
    J_s = 0.50 T_s + 0.22 E_s + 0.13 S_s + 0.10 C_s - 0.22 MS_s - 0.18 P_s
    """
    out = summary_df.copy()

    risk_limit = 0.55
    cost_limit = 0.60
    comp_min = 0.88

    out["FeasibleCompliance"] = out["mean_compliance_reliability"] >= comp_min
    out["FeasibleRisk"] = out["mean_overload_risk"] <= risk_limit
    out["FeasibleCost"] = out["mean_treatment_cost"] <= cost_limit
    out["Feasible"] = out[["FeasibleCompliance", "FeasibleRisk", "FeasibleCost"]].all(axis=1)

    penalty = (
        np.maximum(0.0, out["mean_overload_risk"] - risk_limit) / (1.0 - risk_limit) +
        np.maximum(0.0, comp_min - out["mean_compliance_reliability"]) / comp_min +
        np.maximum(0.0, out["mean_treatment_cost"] - cost_limit) / (1.0 - cost_limit)
    )
    out["ConstraintPenalty"] = penalty

    out["CompositeScore"] = (
        0.50 * out["TopologicalStabilityScore"] +
        0.22 * out["EnvironmentalSafetyScore"] +
        0.13 * out["ServiceCoverageScore"] +
        0.10 * out["EconomicScore"] -
        0.22 * out["MeanSensitivity"] -
        0.18 * out["ConstraintPenalty"]
    )

    out["CompositeRank"] = out["CompositeScore"].rank(ascending=False, method="dense").astype(int)
    out = out.sort_values(["CompositeScore", "Scenario"], ascending=[False, True]).reset_index(drop=True)
    return out


# ======================================================================================
# 7. Projection and visualization
# ======================================================================================

def pca_projection(Z_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, PCA]:
    """
    Compute a two-dimensional projection of the standardized state clouds.

    This projection is only a visual corroboration. All topological calculations are
    performed in the full six-dimensional standardized space.
    """
    scenario_codes: List[str] = []
    rows: List[np.ndarray] = []

    for code in SCENARIOS:
        for row in Z_dict[code]:
            scenario_codes.append(code)
            rows.append(row)

    X = np.vstack(rows)
    pca = PCA(n_components=2, random_state=123)
    proj = pca.fit_transform(X)

    proj_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
    proj_df.insert(0, "Scenario", scenario_codes)
    proj_df["ScenarioName"] = proj_df["Scenario"].map({k: str(v["name"]) for k, v in SCENARIOS.items()})
    return proj_df, pca


def save_beta_curve_plot(curves: Dict[str, pd.DataFrame], beta_col: str, ylabel: str, filepath: Path) -> None:
    """
    Save one figure for beta_0 or beta_1 across all scenarios.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for code, curve_df in curves.items():
        ax.plot(curve_df["eps"], curve_df[beta_col], label=f"{code} – {SCENARIOS[code]['name']}")
    ax.set_xlabel("Filtration threshold ε")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} across wastewater reuse scenarios")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_pca_scatter(proj_df: pd.DataFrame, filepath: Path) -> None:
    """
    Save the low-dimensional corroborative visualization of state clouds.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for code in SCENARIOS:
        sdf = proj_df[proj_df["Scenario"] == code]
        ax.scatter(sdf["PC1"], sdf["PC2"], label=code, alpha=0.70, s=28)

    centers = proj_df.groupby("Scenario")[["PC1", "PC2"]].mean()
    for code, row in centers.iterrows():
        ax.text(row["PC1"], row["PC2"], code, fontsize=9, ha="center", va="center")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("State-cloud projection of simulated reuse scenarios")
    ax.grid(alpha=0.25)
    ax.legend(title="Scenario", fontsize=8)
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_sensitivity_heatmap(sens_detail: pd.DataFrame, filepath: Path) -> None:
    """
    Save the heatmap of perturbation sensitivity.

    Each cell is the mean relative change in compact topological summaries caused by
    one ±10% perturbation in one scenario.
    """
    pivot = sens_detail.copy()
    pivot["Perturbation"] = pivot["Feature"] + np.where(pivot["Change"] > 0, " +10%", " -10%")
    matrix = pivot.pivot_table(index="Scenario", columns="Perturbation", values="AggregateSensitivity")

    desired_order = [
        "Availability -10%",
        "Availability +10%",
        "TreatmentCost -10%",
        "TreatmentCost +10%",
        "OverloadRisk -10%",
        "OverloadRisk +10%",
    ]
    matrix = matrix[desired_order]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Sensitivity of topological summaries to ±10% perturbations")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Aggregate relative change")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_stability_scatter(summary_df: pd.DataFrame, filepath: Path) -> None:
    """
    Save the robustness map relating topological stability to perturbation sensitivity.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(summary_df["MeanSensitivity"], summary_df["TopologicalStabilityScore"], s=70)

    for _, row in summary_df.iterrows():
        ax.text(row["MeanSensitivity"], row["TopologicalStabilityScore"], f" {row['Scenario']}", va="center", fontsize=9)

    ax.set_xlabel("Mean perturbation sensitivity")
    ax.set_ylabel("Topological stability score")
    ax.set_title("Robustness map: stability versus perturbation sensitivity")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_tradeoff_scatter(ranking_df: pd.DataFrame, filepath: Path) -> None:
    """
    Save the compromise-space scatter plot.

    Bubble size is proportional to the economic score so that the figure encodes
    safety, topology, and cost in one compact visual.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    sizes = 80 + 220 * ranking_df["EconomicScore"]
    ax.scatter(ranking_df["EnvironmentalSafetyScore"], ranking_df["TopologicalStabilityScore"], s=sizes, alpha=0.70)

    for _, row in ranking_df.iterrows():
        ax.text(row["EnvironmentalSafetyScore"], row["TopologicalStabilityScore"], f" {row['Scenario']}", va="center", fontsize=9)

    ax.set_xlabel("Environmental safety score")
    ax.set_ylabel("Topological stability score")
    ax.set_title("Compromise space for environmentally constrained scenario selection")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_ranking_bar(ranking_df: pd.DataFrame, filepath: Path) -> None:
    """
    Save the final composite ranking bar chart.

    The rank number is written on top of each bar to make manuscript insertion easier.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(ranking_df["Scenario"], ranking_df["CompositeScore"])
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Composite score")
    ax.set_title("Composite ranking under environmental and topological criteria")
    ax.grid(axis="y", alpha=0.25)

    for i, (_, row) in enumerate(ranking_df.iterrows()):
        ax.text(i, row["CompositeScore"], f"{row['CompositeRank']}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight", dpi=220)
    plt.close(fig)


# ======================================================================================
# 8. Orchestration
# ======================================================================================

def compute_all(n_samples: int = 60, seed: int = 123) -> Dict[str, object]:
    """
    Execute the complete workflow from simulation to final ranking.
    """
    scenario_data, raw_combined = simulate_scenarios(n_samples=n_samples, seed=seed)
    scaler, standardized = standardize_scenarios(scenario_data)

    curves = {code: betti_curve(standardized[code]) for code in SCENARIOS}

    summaries: List[Dict[str, object]] = []
    for code in SCENARIOS:
        topo_summary = summarize_curve(curves[code])
        env_summary = baseline_environmental_metrics(scenario_dataframe(code, scenario_data[code]))
        summaries.append({
            "Scenario": code,
            "ScenarioName": str(SCENARIOS[code]["name"]),
            **topo_summary,
            **env_summary,
        })

    summary_df = pd.DataFrame(summaries)

    sens_detail, sens_agg, baseline_curves, baseline_summaries = sensitivity_analysis(scenario_data)
    assert set(curves.keys()) == set(baseline_curves.keys())

    summary_df = summary_df.merge(sens_agg, on=["Scenario", "ScenarioName"], how="left")
    summary_df = compute_indices(summary_df)
    ranking_df = compute_ranking(summary_df)
    projection_df, pca_model = pca_projection(standardized)

    return {
        "scenario_data": scenario_data,
        "raw_combined": raw_combined,
        "scaler": scaler,
        "standardized": standardized,
        "curves": curves,
        "summary_df": summary_df,
        "sens_detail": sens_detail,
        "sens_agg": sens_agg,
        "ranking_df": ranking_df,
        "projection_df": projection_df,
        "pca": pca_model,
    }


def export_results(results: Dict[str, object], directories: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Export all manuscript-ready files.
    """
    base_dir = directories["base_dir"]
    dir_51 = directories["dir_51"]
    dir_52 = directories["dir_52"]
    dir_53 = directories["dir_53"]
    curves_dir = directories["curves_dir"]

    # ------------------------------------------------------------------
    # Global state export
    # ------------------------------------------------------------------
    write_dataframe(results["raw_combined"], base_dir / "simulated_states.csv")

    # ------------------------------------------------------------------
    # Section 5.1 outputs
    # ------------------------------------------------------------------
    save_beta_curve_plot(results["curves"], "beta0", "Betti-0", dir_51 / "Figure_1_beta0_curves.png")
    save_beta_curve_plot(results["curves"], "beta1", "Betti-1", dir_51 / "Figure_2_beta1_curves.png")
    save_pca_scatter(results["projection_df"], dir_51 / "Figure_3_state_cloud_projection.png")

    table1 = (
        results["summary_df"][[
            "Scenario",
            "ScenarioName",
            "AUBC0",
            "AUBC1",
            "eps_conn",
            "eps_loop",
            "peak_beta1",
            "mean_availability",
            "mean_treatment_cost",
            "mean_overload_risk",
            "mean_compliance_reliability",
            "mean_demand_coverage",
        ]]
        .copy()
        .sort_values("Scenario")
        .reset_index(drop=True)
    )

    write_dataframe(
        table1.round(4),
        dir_51 / "Table_1_topological_summary.csv",
        dir_51 / "Table_1_topological_summary.xlsx",
    )

    # ------------------------------------------------------------------
    # Section 5.2 outputs
    # ------------------------------------------------------------------
    save_sensitivity_heatmap(results["sens_detail"], dir_52 / "Figure_4_sensitivity_heatmap.png")
    save_stability_scatter(results["ranking_df"], dir_52 / "Figure_5_stability_vs_sensitivity.png")

    worst = results["sens_detail"].copy()
    worst["Perturbation"] = worst["Feature"] + np.where(worst["Change"] > 0, " +10%", " -10%")
    worst = worst.loc[
        worst.groupby("Scenario")["AggregateSensitivity"].idxmax(),
        ["Scenario", "ScenarioName", "Perturbation", "AggregateSensitivity"],
    ]

    table2 = (
        results["ranking_df"][[
            "CompositeRank",
            "Scenario",
            "ScenarioName",
            "TopologicalStabilityScore",
            "MeanSensitivity",
            "MeanAUBC0Sensitivity",
            "MeanAUBC1Sensitivity",
            "MeanConnectionSensitivity",
            "MeanLoopPeakSensitivity",
        ]]
        .merge(worst, on=["Scenario", "ScenarioName"], how="left")
        .sort_values("CompositeRank")
        .reset_index(drop=True)
    )

    write_dataframe(
        table2.round(4),
        dir_52 / "Table_2_robustness_sensitivity.csv",
        dir_52 / "Table_2_robustness_sensitivity.xlsx",
    )

    # ------------------------------------------------------------------
    # Section 5.3 outputs
    # ------------------------------------------------------------------
    save_tradeoff_scatter(results["ranking_df"], dir_53 / "Figure_6_tradeoff_space.png")
    save_ranking_bar(results["ranking_df"], dir_53 / "Figure_7_composite_ranking.png")

    table3 = (
        results["ranking_df"][[
            "CompositeRank",
            "Scenario",
            "ScenarioName",
            "Feasible",
            "ConstraintPenalty",
            "TopologicalStabilityScore",
            "EnvironmentalSafetyScore",
            "ServiceCoverageScore",
            "EconomicScore",
            "MeanSensitivity",
            "CompositeScore",
        ]]
        .copy()
        .sort_values("CompositeRank")
        .reset_index(drop=True)
    )

    write_dataframe(
        table3.round(4),
        dir_53 / "Table_3_composite_ranking.csv",
        dir_53 / "Table_3_composite_ranking.xlsx",
    )

    # ------------------------------------------------------------------
    # Curve exports
    # ------------------------------------------------------------------
    for code, curve_df in results["curves"].items():
        write_dataframe(curve_df.round(6), curves_dir / f"{code}_betti_curve.csv")

    return {
        "table1": table1.round(4),
        "table2": table2.round(4),
        "table3": table3.round(4),
    }


def save_run_summary(results: Dict[str, object], directories: Dict[str, Path]) -> None:
    """
    Write a compact narrative summary of the main findings.

    This file is useful when the script is executed outside the notebook or when the
    author wants a quick textual check before opening the tables and figures.
    """
    ranking_df = results["ranking_df"].copy()
    summary_df = results["summary_df"].copy()

    best = ranking_df.iloc[0]
    runner_up = ranking_df.iloc[1]
    weakest = ranking_df.iloc[-1]

    lines = [
        "Betti-Curve Analysis of Wastewater Reuse Scenarios",
        "=" * 55,
        "",
        f"Number of scenarios: {len(SCENARIOS)}",
        f"States per scenario: {len(next(iter(results['scenario_data'].values())))}",
        "",
        "Top-ranked scenario:",
        f"  {best['Scenario']} - {best['ScenarioName']}",
        f"  Composite score: {best['CompositeScore']:.4f}",
        f"  Topological stability: {best['TopologicalStabilityScore']:.4f}",
        f"  Mean sensitivity: {best['MeanSensitivity']:.4f}",
        "",
        "Second-ranked scenario:",
        f"  {runner_up['Scenario']} - {runner_up['ScenarioName']}",
        f"  Composite score: {runner_up['CompositeScore']:.4f}",
        "",
        "Weakest scenario:",
        f"  {weakest['Scenario']} - {weakest['ScenarioName']}",
        f"  Composite score: {weakest['CompositeScore']:.4f}",
        "",
        "Connectivity thresholds (eps_conn):",
    ]

    for _, row in summary_df.sort_values("eps_conn").iterrows():
        lines.append(f"  {row['Scenario']}: {row['eps_conn']:.4f}")

    lines.extend([
        "",
        "Interpretive note:",
        "  Lower AUBC0 and earlier eps_conn indicate greater compactness and faster",
        "  structural consolidation of the scenario cloud under the filtration.",
        "",
    ])

    (directories["base_dir"] / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ======================================================================================
# 9. Command-line interface
# ======================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Betti-curve-based analysis for wastewater reuse scenarios."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wastewater_reuse_betti_results",
        help="Base directory where results/5.1, results/5.2, and results/5.3 will be created.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=60,
        help="Number of simulated states per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for the multivariate scenario simulation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.output_dir).resolve()
    directories = ensure_dirs(base_dir)

    results = compute_all(n_samples=args.n_samples, seed=args.seed)
    exported_tables = export_results(results, directories)
    save_run_summary(results, directories)

    # Console summary for quick external execution feedback.
    ranking = results["ranking_df"][[
        "CompositeRank",
        "Scenario",
        "ScenarioName",
        "Feasible",
        "TopologicalStabilityScore",
        "EnvironmentalSafetyScore",
        "ServiceCoverageScore",
        "EconomicScore",
        "CompositeScore",
    ]].copy()

    print("\nTop-ranked scenarios")
    print("--------------------")
    print(ranking.to_string(index=False))

    print("\nOutput directory")
    print("----------------")
    print(base_dir)

    print("\nManuscript-ready folders created")
    print("--------------------------------")
    print(base_dir / "results" / "5.1")
    print(base_dir / "results" / "5.2")
    print(base_dir / "results" / "5.3")


if __name__ == "__main__":
    main()
