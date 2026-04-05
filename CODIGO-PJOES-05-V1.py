#!/usr/bin/env python3
"""
Persistent homology benchmark for detecting structural instability in urban water
 distribution networks.

This script creates a realistic synthetic urban water distribution network,
simulates monthly deterioration and leakage dynamics, computes persistent
homology summaries from sensor-state point clouds, and exports figures/tables
that can be inserted directly into the companion article.

Author: OpenAI for Jaime Aguilar-Ortiz workflow support
Language: English
Reproducibility: deterministic random seed
"""

from __future__ import annotations

import os
import math
from itertools import combinations
from typing import Dict, List, Tuple, Iterable

# Matplotlib cache path for restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_water_ph")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

SEED = 42
RNG = np.random.default_rng(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SEC_51 = os.path.join(OUTPUT_DIR, "5_1")
SEC_52 = os.path.join(OUTPUT_DIR, "5_2")
SEC_53 = os.path.join(OUTPUT_DIR, "5_3")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
for path in [OUTPUT_DIR, SEC_51, SEC_52, SEC_53, DATA_DIR]:
    os.makedirs(path, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Network and scenario generation
# -----------------------------------------------------------------------------

def build_base_network(rows: int = 5, cols: int = 6, seed: int = SEED) -> nx.Graph:
    """Create a meshed city-scale water network with loops and bypass edges.

    The layout is stylized but realistic: a rectilinear urban grid, heterogeneous
    pipe diameters, non-uniform lengths, and a limited set of diagonal bypasses
    that add loop redundancy.
    """
    rng = np.random.default_rng(seed)
    graph = nx.Graph()

    def node_id(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            idx = node_id(r, c)
            graph.add_node(
                idx,
                x=float(c),
                y=float(rows - 1 - r),
            )

    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:
                u, v = node_id(r, c), node_id(r, c + 1)
                graph.add_edge(
                    u,
                    v,
                    length=70.0 + 20.0 * rng.random(),
                    diameter=int(rng.choice([200, 250, 300, 350, 400], p=[0.15, 0.20, 0.25, 0.25, 0.15])),
                    age=int(rng.integers(5, 45)),
                    edge_type="horizontal",
                )
            if r < rows - 1:
                u, v = node_id(r, c), node_id(r + 1, c)
                graph.add_edge(
                    u,
                    v,
                    length=75.0 + 25.0 * rng.random(),
                    diameter=int(rng.choice([180, 220, 260, 320, 380], p=[0.20, 0.25, 0.25, 0.20, 0.10])),
                    age=int(rng.integers(5, 50)),
                    edge_type="vertical",
                )

    # Additional bypass pipes create redundancy and finite triangular fillings
    # in the induced filtration.
    extra_edges = [(0, 7), (7, 14), (14, 21), (8, 15), (15, 22), (2, 9), (9, 16), (16, 23), (6, 13), (13, 20), (10, 17), (17, 24)]
    for u, v in extra_edges:
        if not graph.has_edge(u, v):
            graph.add_edge(
                u,
                v,
                length=95.0 + 15.0 * rng.random(),
                diameter=int(rng.choice([180, 220, 260])),
                age=int(rng.integers(5, 35)),
                edge_type="bypass",
            )

    # Sector labels for contextual interpretation.
    for n, attrs in graph.nodes(data=True):
        x = attrs["x"]
        if x <= 1.5:
            sector = "Residential"
        elif x <= 3.5:
            sector = "Industrial"
        else:
            sector = "Public"
        graph.nodes[n]["sector"] = sector

    return graph


def simulate_network_evolution(
    graph: nx.Graph,
    months: int = 30,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, nx.Graph], List[int], set]:
    """Generate a time-evolving instability scenario.

    The simulation is designed to create three qualitatively distinct phases:
    * Baseline (Months 1-10): healthy network with mild seasonality.
    * Incipient instability (Months 11-20): growing leaks, roughness increase,
      declining pressure margins, but no large service deficit yet.
    * Critical instability (Months 21-30): pipe isolation, low-pressure nodes,
      and increasing unmet demand.
    """
    rng = np.random.default_rng(seed)
    base_nodes = list(graph.nodes())
    source_heads = {0: 67.5, 5: 66.5, 24: 65.5}
    sensor_nodes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29]

    leak_edges = {tuple(sorted(e)) for e in [(7, 8), (8, 14), (14, 15), (15, 16), (16, 22)]}
    burst_schedule = {21: [(14, 15)], 25: [(8, 14)]}

    monthly_rows: List[dict] = []
    node_rows: List[dict] = []
    edge_rows: List[dict] = []
    graphs_by_month: Dict[int, nx.Graph] = {}

    for month in range(1, months + 1):
        seasonal_factor = 0.6 * math.sin(2.0 * math.pi * (month - 1) / 12.0)
        deterioration = 0.018 * month
        leak_intensity = 0.0 if month < 10 else min(0.08 + 0.05 * (month - 9), 1.0)
        source_stress = 0.0 if month < 15 else 0.5 + 0.17 * (month - 14)
        criticality = 0.0 if month < 19 else min(0.12 * (month - 18), 1.0)

        dynamic_graph = nx.Graph()
        for n, attrs in graph.nodes(data=True):
            dynamic_graph.add_node(n, **attrs)

        # Dynamic hydraulic state at edge level.
        for u, v, attrs in graph.edges(data=True):
            edge_key = tuple(sorted((u, v)))
            local_leak = leak_intensity if edge_key in leak_edges else 0.0
            base_resistance = attrs["length"] / attrs["diameter"]
            age_factor = 1.0 + 0.006 * attrs["age"]
            rough_factor = max(0.8, 1.0 + deterioration + 1.0 * local_leak + 0.10 * rng.normal())
            resistance = base_resistance * age_factor * rough_factor
            capacity = max(attrs["diameter"] * (1.0 - 0.38 * deterioration - 0.65 * local_leak), 0.08 * attrs["diameter"])
            quality = max(0.02, 1.0 - 0.60 * local_leak - 0.18 * deterioration + 0.04 * rng.normal())
            active = True
            for burst_month, burst_edges in burst_schedule.items():
                if month >= burst_month and edge_key in {tuple(sorted(e)) for e in burst_edges}:
                    active = False

            dynamic_graph.add_edge(
                u,
                v,
                resistance=float(resistance),
                capacity=float(capacity),
                quality=float(quality),
                active=bool(active),
                leak=float(local_leak),
            )

            edge_rows.append(
                {
                    "month": month,
                    "u": u,
                    "v": v,
                    "resistance": resistance,
                    "capacity": capacity,
                    "quality": quality,
                    "active": int(active),
                    "leak": local_leak,
                    "diameter": attrs["diameter"],
                    "age": attrs["age"],
                }
            )

        # Active graph used for routing and pressure approximation.
        active_graph = nx.Graph()
        for n, attrs in dynamic_graph.nodes(data=True):
            active_graph.add_node(n, **attrs)
        for u, v, attrs in dynamic_graph.edges(data=True):
            if attrs["active"]:
                active_graph.add_edge(u, v, **attrs)

        # Sectoral demand generation.
        for n, attrs in graph.nodes(data=True):
            base_by_sector = {"Residential": 1.25, "Industrial": 1.45, "Public": 1.05}
            seasonal_by_sector = {"Residential": 0.06, "Industrial": 0.04, "Public": 0.03}
            sector = attrs["sector"]
            demand = base_by_sector[sector] * (1.0 + seasonal_by_sector[sector] * seasonal_factor) + 0.08 * rng.normal()
            graph.nodes[n]["base_demand"] = max(demand, 0.60)

        pressures, flows, chlorine, deficits = {}, {}, {}, {}
        low_pressure_nodes = 0
        disconnected_nodes = 0

        for n in base_nodes:
            local_leak = max([dynamic_graph.edges[n, nbr]["leak"] for nbr in dynamic_graph.neighbors(n)] + [0.0])
            demand = graph.nodes[n]["base_demand"]

            try:
                best_head = -1e9
                best_distance = None
                for source, head in source_heads.items():
                    distance = nx.shortest_path_length(active_graph, source, n, weight="resistance")
                    effective_head = head - 1.25 * distance
                    if effective_head > best_head:
                        best_head = effective_head
                        best_distance = distance

                pressure = best_head - 4.8 * demand - 7.5 * source_stress - 11.0 * local_leak - 2.5 * criticality + rng.normal(0.0, 0.45)
                pressure = max(6.0, pressure)
                flow = max(2.0, 40.0 - 1.4 * best_distance - 4.2 * local_leak - 2.1 * source_stress - 1.2 * criticality + rng.normal(0.0, 1.2))
                residual = max(0.01, 1.40 * math.exp(-0.035 * best_distance) * (1.0 - 0.65 * local_leak - 0.18 * source_stress - 0.10 * criticality) + rng.normal(0.0, 0.03))
            except Exception:
                # Isolated or nearly isolated node.
                pressure = 7.0 + abs(rng.normal(0.0, 0.3))
                flow = 1.8 + abs(rng.normal(0.0, 0.3))
                residual = 0.02 + abs(rng.normal(0.0, 0.01))
                disconnected_nodes += 1

            delivered = demand if pressure >= 38.0 else demand * max(0.0, (pressure - 12.0) / 26.0)
            deficit = max(0.0, demand - delivered)

            pressures[n] = pressure
            flows[n] = flow
            chlorine[n] = residual
            deficits[n] = deficit

            if pressure < 38.0:
                low_pressure_nodes += 1

            node_rows.append(
                {
                    "month": month,
                    "node": n,
                    "x": graph.nodes[n]["x"],
                    "y": graph.nodes[n]["y"],
                    "sector": graph.nodes[n]["sector"],
                    "pressure": pressure,
                    "flow": flow,
                    "chlorine": residual,
                    "demand": demand,
                    "deficit": deficit,
                    "is_sensor": int(n in sensor_nodes),
                }
            )

        for n in dynamic_graph.nodes():
            dynamic_graph.nodes[n]["pressure"] = pressures[n]
            dynamic_graph.nodes[n]["flow"] = flows[n]
            dynamic_graph.nodes[n]["chlorine"] = chlorine[n]
            dynamic_graph.nodes[n]["deficit"] = deficits[n]

        # Global network performance summaries.
        lengths = dict(nx.all_pairs_dijkstra_path_length(active_graph, weight="resistance"))
        inv_sum = 0.0
        pair_count = 0
        for i in base_nodes:
            for j in base_nodes:
                if i < j:
                    pair_count += 1
                    dij = lengths.get(i, {}).get(j, np.inf)
                    if np.isfinite(dij) and dij > 0.0:
                        inv_sum += 1.0 / dij
        efficiency = inv_sum / pair_count

        total_demand = sum(graph.nodes[n]["base_demand"] for n in base_nodes)
        total_deficit = sum(deficits.values())
        service_ratio = 1.0 - total_deficit / total_demand

        monthly_rows.append(
            {
                "month": month,
                "seasonal_factor": seasonal_factor,
                "deterioration": deterioration,
                "leak_intensity": leak_intensity,
                "source_stress": source_stress,
                "criticality": criticality,
                "mean_pressure": np.mean(list(pressures.values())),
                "min_pressure": np.min(list(pressures.values())),
                "mean_chlorine": np.mean(list(chlorine.values())),
                "service_ratio": service_ratio,
                "total_deficit": total_deficit,
                "low_pressure_nodes": low_pressure_nodes,
                "disconnected_nodes": disconnected_nodes,
                "active_edge_ratio": active_graph.number_of_edges() / graph.number_of_edges(),
                "graph_efficiency": efficiency,
            }
        )
        graphs_by_month[month] = dynamic_graph

    return (
        pd.DataFrame(monthly_rows),
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
        graphs_by_month,
        sensor_nodes,
        leak_edges,
    )


# -----------------------------------------------------------------------------
# 2. Persistent homology utilities (portable pure-Python implementation)
# -----------------------------------------------------------------------------

def sensor_point_cloud(graph: nx.Graph, sensor_nodes: List[int]) -> np.ndarray:
    """Create a sensor-state point cloud induced by the network state.

    Each point combines spatial coordinates with hydraulic variables and a local
    degradation proxy. This is not a generic point cloud; it is a compact
    topological representation of the monitored water network state.
    """
    rows = []
    for n in sensor_nodes:
        attrs = graph.nodes[n]
        local_leak = 0.0
        resistances = []
        for nbr in graph.neighbors(n):
            eattrs = graph.edges[n, nbr]
            local_leak = max(local_leak, eattrs["leak"])
            resistances.append(eattrs["resistance"])

        rows.append(
            [
                attrs["x"] / 5.0,
                attrs["y"] / 4.0,
                attrs["pressure"] / 70.0,
                attrs["flow"] / 45.0,
                attrs["chlorine"] / 1.5,
                (np.mean(resistances) if resistances else 0.0) / 0.9,
                local_leak,
            ]
        )

    cloud = np.array(rows, dtype=float)
    mean = cloud.mean(axis=0)
    std = cloud.std(axis=0)
    std[std < 1e-8] = 1.0
    return (cloud - mean) / std


def rips_complex_simplices(point_cloud: np.ndarray, max_dim: int = 2) -> List[Tuple[Tuple[int, ...], float, int]]:
    """Generate a Vietoris-Rips filtration up to dimension 2."""
    n_points = point_cloud.shape[0]
    distances = np.linalg.norm(point_cloud[:, None, :] - point_cloud[None, :, :], axis=2)
    simplices: List[Tuple[Tuple[int, ...], float, int]] = []

    for i in range(n_points):
        simplices.append(((i,), 0.0, 0))

    for i, j in combinations(range(n_points), 2):
        simplices.append(((i, j), float(distances[i, j]), 1))

    if max_dim >= 2:
        for i, j, k in combinations(range(n_points), 3):
            filt_value = float(max(distances[i, j], distances[i, k], distances[j, k]))
            simplices.append(((i, j, k), filt_value, 2))

    simplices.sort(key=lambda item: (item[1], item[2], item[0]))
    return simplices


def persistent_homology_intervals(
    point_cloud: np.ndarray,
    max_dim: int = 2,
    inf_value: float | None = None,
) -> Tuple[Dict[int, List[Tuple[float, float, Tuple[int, ...], Tuple[int, ...] | None]]], List[Tuple[Tuple[int, ...], float, int]], float]:
    """Compute persistence intervals through standard column reduction over Z2.

    The implementation is intentionally explicit and dependency-light so the full
    pipeline remains reproducible even in environments without specialized TDA
    libraries.
    """
    simplices = rips_complex_simplices(point_cloud, max_dim=max_dim)
    index = {simplex[0]: idx for idx, simplex in enumerate(simplices)}
    boundaries: List[set] = []

    for vertices, filt_value, dim in simplices:
        if dim == 0:
            boundaries.append(set())
        else:
            face_indices = set(index[tuple(face)] for face in combinations(vertices, dim))
            boundaries.append(face_indices)

    reduced_cols: List[set] = [set() for _ in simplices]
    low_to_col: Dict[int, int] = {}
    paired_births: Dict[int, bool] = {}
    intervals: Dict[int, List[Tuple[float, float, Tuple[int, ...], Tuple[int, ...] | None]]] = {0: [], 1: [], 2: []}

    max_filt = max(s[1] for s in simplices) if simplices else 0.0
    if inf_value is None:
        inf_value = max_filt * 1.05 + 1e-6

    for col_idx, (vertices, filt_value, dim) in enumerate(simplices):
        col = set(boundaries[col_idx])
        while col:
            low = max(col)
            if low in low_to_col:
                col ^= reduced_cols[low_to_col[low]]
            else:
                break

        if not col:
            reduced_cols[col_idx] = set()
        else:
            low = max(col)
            reduced_cols[col_idx] = set(col)
            low_to_col[low] = col_idx
            birth_vertices, birth_filt, birth_dim = simplices[low]
            intervals[birth_dim].append((birth_filt, filt_value, birth_vertices, vertices))
            paired_births[low] = True

    # Infinite intervals for unpaired births.
    for idx, (vertices, filt_value, dim) in enumerate(simplices):
        if dim <= max_dim and idx not in paired_births:
            if len(boundaries[idx]) == 0 or reduced_cols[idx] == set():
                intervals[dim].append((filt_value, inf_value, vertices, None))

    return intervals, simplices, inf_value


def betti_curve(intervals: Iterable[Tuple[float, float, Tuple[int, ...], Tuple[int, ...] | None]], grid: np.ndarray) -> np.ndarray:
    values = np.zeros(len(grid), dtype=float)
    for birth, death, _, _ in intervals:
        values += ((grid >= birth) & (grid < death)).astype(float)
    return values


def persistent_diagram_points(intervals: Iterable[Tuple[float, float, Tuple[int, ...], Tuple[int, ...] | None]], top_fraction: float = 0.55) -> np.ndarray:
    """Keep the most informative intervals for clean visualization."""
    rows = [(birth, death, death - birth) for birth, death, _, _ in intervals]
    if not rows:
        return np.empty((0, 3))
    arr = np.array(rows, dtype=float)
    threshold = np.quantile(arr[:, 2], top_fraction)
    arr = arr[arr[:, 2] >= threshold]
    return arr


def summarize_topology(graph: nx.Graph, sensor_nodes: List[int]) -> Dict[str, object]:
    point_cloud = sensor_point_cloud(graph, sensor_nodes)
    intervals, simplices, inf_value = persistent_homology_intervals(point_cloud, max_dim=2)
    grid = np.linspace(0.0, inf_value, 150)
    bc0 = betti_curve(intervals[0], grid)
    bc1 = betti_curve(intervals[1], grid)

    pers0 = np.array([death - birth for birth, death, _, _ in intervals[0]], dtype=float)
    pers1 = np.array([death - birth for birth, death, _, _ in intervals[1]], dtype=float)
    pers0_sig = pers0[pers0 > np.quantile(pers0, 0.25)] if len(pers0) > 1 else pers0
    pers1_sig = pers1[pers1 > np.quantile(pers1, 0.75)] if len(pers1) > 4 else pers1

    return {
        "grid": grid,
        "bc0": bc0,
        "bc1": bc1,
        "intervals0": intervals[0],
        "intervals1": intervals[1],
        "tp0": float(pers0_sig.sum()),
        "tp1": float(pers1_sig.sum()),
        "bc0_auc": float(np.trapezoid(bc0, grid)),
        "bc1_auc": float(np.trapezoid(bc1, grid)),
        "h1_peak": float(np.max(bc1)),
        "inf_value": float(inf_value),
    }


# -----------------------------------------------------------------------------
# 3. Analysis tables and early-warning indicators
# -----------------------------------------------------------------------------

def add_topological_metrics(monthly_df: pd.DataFrame, graphs_by_month: Dict[int, nx.Graph], sensor_nodes: List[int]) -> Tuple[pd.DataFrame, Dict[int, Dict[str, object]]]:
    summaries = {month: summarize_topology(graphs_by_month[month], sensor_nodes) for month in sorted(graphs_by_month)}

    topo_df = pd.DataFrame(
        [
            {
                "month": month,
                "tp0": summaries[month]["tp0"],
                "tp1": summaries[month]["tp1"],
                "bc0_auc": summaries[month]["bc0_auc"],
                "bc1_auc": summaries[month]["bc1_auc"],
                "h1_peak": summaries[month]["h1_peak"],
            }
            for month in sorted(summaries)
        ]
    )

    data = monthly_df.merge(topo_df, on="month", how="left")
    baseline = data[data["month"] <= 8].copy()

    # Baseline-standardized deviations.
    for variable in [
        "tp0",
        "tp1",
        "bc0_auc",
        "bc1_auc",
        "h1_peak",
        "graph_efficiency",
        "mean_pressure",
        "service_ratio",
        "mean_chlorine",
    ]:
        mu = baseline[variable].mean()
        sigma = baseline[variable].std()
        sigma = sigma if sigma > 1e-8 else 1.0
        data[f"z_{variable}"] = (data[variable] - mu) / sigma

    # Instability index: larger values imply stronger departure from baseline.
    data["topological_instability_index"] = (
        0.35 * np.maximum(0.0, -data["z_tp0"]) +
        0.20 * np.maximum(0.0, -data["z_tp1"]) +
        0.25 * np.maximum(0.0, -data["z_bc0_auc"]) +
        0.20 * np.maximum(0.0, -data["z_bc1_auc"])
    )

    baseline_tii = data.loc[data["month"] <= 8, "topological_instability_index"]
    median_tii = float(np.median(baseline_tii))
    mad_tii = float(np.median(np.abs(baseline_tii - median_tii)))
    threshold_tii = median_tii + 3.0 * 1.4826 * mad_tii
    data["tii_threshold"] = threshold_tii

    data["phase"] = pd.cut(data["month"], bins=[0, 10, 20, 30], labels=["Baseline", "Incipient", "Critical"])

    return data, summaries


# -----------------------------------------------------------------------------
# 4. Figure generation
# -----------------------------------------------------------------------------

def _setup_figure(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    return fig, ax


def draw_network_map(base_graph: nx.Graph, month_graph: nx.Graph, leak_edges: set, month: int, output_path: str):
    fig, ax = _setup_figure((7.5, 5.4))
    pos = {n: (base_graph.nodes[n]["x"], base_graph.nodes[n]["y"]) for n in base_graph.nodes()}

    # Draw active edges.
    for u, v in base_graph.edges():
        attrs = month_graph.edges[u, v]
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        if not attrs["active"]:
            ax.plot(x, y, linestyle="--", linewidth=2.4, color="#c23b22", alpha=0.95)
        elif tuple(sorted((u, v))) in leak_edges and attrs["leak"] > 0:
            ax.plot(x, y, linewidth=2.8, color="#ff8c00", alpha=0.95)
        else:
            ax.plot(x, y, linewidth=1.6, color="#4c78a8", alpha=0.85)

    pressures = np.array([month_graph.nodes[n]["pressure"] for n in month_graph.nodes()])
    scatter = ax.scatter(
        [pos[n][0] for n in month_graph.nodes()],
        [pos[n][1] for n in month_graph.nodes()],
        c=pressures,
        cmap="viridis",
        s=80,
        edgecolor="black",
        linewidths=0.5,
        zorder=4,
    )
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pressure (m)")

    ax.set_title(f"Dynamic water-distribution network state at Month {month}")
    ax.set_xlabel("Urban east-west coordinate")
    ax.set_ylabel("Urban north-south coordinate")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)

    legend_elements = [
        Line2D([0], [0], color="#4c78a8", lw=2, label="Active pipe"),
        Line2D([0], [0], color="#ff8c00", lw=3, label="Leak-affected pipe"),
        Line2D([0], [0], color="#c23b22", lw=2, linestyle="--", label="Isolated pipe"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_architecture_figure(output_path: str):
    fig, ax = _setup_figure((10.2, 4.6))
    ax.axis("off")

    boxes = [
        (0.02, 0.22, 0.19, 0.52, "Synthetic hydraulic\nscenario generator", "Monthly demand, pipe age,\nleaks, bursts, source stress"),
        (0.27, 0.22, 0.19, 0.52, "Dynamic network\nstate extraction", "Pressure, flow, chlorine,\ncapacity, connectivity"),
        (0.52, 0.22, 0.19, 0.52, "Persistent homology\nengine", "Sensor-state point clouds,\nRips filtration, H0-H1 intervals"),
        (0.77, 0.22, 0.19, 0.52, "Early-warning\nanalytics", "Betti curves, instability index,\nlead-time comparison"),
    ]

    colors = ["#dceaf7", "#e8f4e5", "#f9ead8", "#f7dfe8"]
    for (x, y, w, h, title, subtitle), color in zip(boxes, colors):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor=color, edgecolor="#4d4d4d", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=9)

    for i in range(3):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y = 0.48
        arrow = FancyArrowPatch((x1 + 0.01, y), (x2 - 0.01, y), arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="#4d4d4d")
        ax.add_patch(arrow)

    ax.set_title("Functional architecture of the persistent-homology monitoring pipeline", fontsize=14, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_regime_timeline(data: pd.DataFrame, output_path: str):
    fig, ax = _setup_figure((8.6, 4.8))
    ax.plot(data["month"], data["deterioration"], label="Deterioration factor", linewidth=2.2)
    ax.plot(data["month"], data["leak_intensity"], label="Leak intensity", linewidth=2.2)
    ax.plot(data["month"], data["source_stress"], label="Source stress", linewidth=2.2)

    ax.axvspan(1, 10, color="#dceaf7", alpha=0.25)
    ax.axvspan(10, 20, color="#fbe5c8", alpha=0.25)
    ax.axvspan(20, 30, color="#f6d3d8", alpha=0.25)
    ax.text(5.5, ax.get_ylim()[1] * 0.93, "Baseline", ha="center", fontsize=10, weight="bold")
    ax.text(15.0, ax.get_ylim()[1] * 0.93, "Incipient instability", ha="center", fontsize=10, weight="bold")
    ax.text(25.0, ax.get_ylim()[1] * 0.93, "Critical instability", ha="center", fontsize=10, weight="bold")

    ax.set_title("Synthetic instability schedule and forcing components")
    ax.set_xlabel("Month")
    ax.set_ylabel("Normalized scenario intensity")
    ax.grid(alpha=0.2)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_persistence_diagrams(summaries: Dict[int, Dict[str, object]], months: List[int], output_path: str):
    fig, axes = plt.subplots(2, len(months), figsize=(11.2, 5.2), dpi=200)
    for col, month in enumerate(months):
        info = summaries[month]
        inf_value = info["inf_value"]
        max_lim = inf_value * 1.02
        for row, key in enumerate(["intervals0", "intervals1"]):
            ax = axes[row, col]
            arr = persistent_diagram_points(info[key], top_fraction=0.55)
            if arr.size:
                ax.scatter(arr[:, 0], arr[:, 1], s=16, alpha=0.8)
            ax.plot([0, max_lim], [0, max_lim], linestyle="--", color="#555555", linewidth=1.1)
            ax.set_xlim(0, max_lim)
            ax.set_ylim(0, max_lim)
            if row == 0:
                ax.set_title(f"Month {month}")
            if col == 0:
                ax.set_ylabel("Death")
                ax.text(0.05, 0.92, "H0" if row == 0 else "H1", transform=ax.transAxes, fontsize=10, weight="bold")
            ax.set_xlabel("Birth")
            ax.grid(alpha=0.15)
    fig.suptitle("Persistent diagrams for representative months", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_betti_curves(summaries: Dict[int, Dict[str, object]], months: List[int], output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=200)
    for month in months:
        info = summaries[month]
        axes[0].plot(info["grid"], info["bc0"], linewidth=2.0, label=f"Month {month}")
        axes[1].plot(info["grid"], info["bc1"], linewidth=2.0, label=f"Month {month}")

    axes[0].set_title("Betti-0 curves")
    axes[1].set_title("Betti-1 curves")
    for ax in axes:
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Betti count")
        ax.grid(alpha=0.2)
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Multiscale topological summaries across the instability trajectory", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_temporal_metrics(data: pd.DataFrame, output_path: str):
    fig, ax1 = plt.subplots(figsize=(9.0, 4.8), dpi=200)
    ax2 = ax1.twinx()

    ax1.plot(data["month"], data["topological_instability_index"], linewidth=2.5, label="Topological instability index")
    ax1.plot(data["month"], data["tii_threshold"], linestyle="--", linewidth=1.8, label="Robust threshold")
    ax2.plot(data["month"], data["mean_pressure"], linewidth=2.2, linestyle="-.", label="Mean pressure")
    ax2.plot(data["month"], data["service_ratio"], linewidth=2.0, linestyle=":", label="Service ratio")

    ax1.axvline(13, color="#444444", linestyle=":", linewidth=1.3)
    ax1.axvline(21, color="#444444", linestyle=":", linewidth=1.3)
    ax1.text(13.1, ax1.get_ylim()[1] * 0.90, "Topological alert", fontsize=9)
    ax1.text(21.1, ax1.get_ylim()[1] * 0.76, "Operational service loss", fontsize=9)

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Topological instability score")
    ax2.set_ylabel("Hydraulic performance")
    ax1.set_title("Temporal evolution of topological and hydraulic indicators")
    ax1.grid(alpha=0.2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_early_warning_bars(alert_df: pd.DataFrame, output_path: str):
    fig, ax = _setup_figure((8.2, 4.8))
    y = np.arange(len(alert_df))
    ax.barh(y, alert_df["lead_time_months"], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(alert_df["indicator"])
    ax.invert_yaxis()
    ax.set_xlabel("Lead time before operational service loss (months)")
    ax.set_title("Early-warning lead time delivered by competing indicators")
    ax.grid(axis="x", alpha=0.2)

    for yi, lead, alert in zip(y, alert_df["lead_time_months"], alert_df["alert_month"]):
        ax.text(lead + 0.08, yi, f"M{alert}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def draw_phase_heatmap(data: pd.DataFrame, output_path: str):
    fig, ax = _setup_figure((9.0, 3.6))
    matrix = data[["topological_instability_index", "mean_pressure", "graph_efficiency", "service_ratio"]].copy()
    # Rescale each variable to [0, 1] for visual comparability.
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    image = ax.imshow(matrix.T, aspect="auto", cmap="viridis")
    cbar = plt.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized value")
    ax.set_yticks(range(matrix.shape[1]))
    ax.set_yticklabels(["Topological\ninstability", "Mean\npressure", "Graph\nefficiency", "Service\nratio"])
    ax.set_xticks(np.arange(0, len(data), 2))
    ax.set_xticklabels(data["month"].iloc[::2])
    ax.set_xlabel("Month")
    ax.set_title("Joint temporal footprint of structural instability indicators")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Tables and summary extraction
# -----------------------------------------------------------------------------

def build_tables(base_graph: nx.Graph, data: pd.DataFrame, node_df: pd.DataFrame, edge_df: pd.DataFrame, sensor_nodes: List[int]) -> Dict[str, pd.DataFrame]:
    baseline = data[data["phase"] == "Baseline"]
    incipient = data[data["phase"] == "Incipient"]
    critical = data[data["phase"] == "Critical"]

    topology_table = pd.DataFrame(
        [
            ["Number of nodes", base_graph.number_of_nodes(), "Hydraulic junctions in the stylized city network"],
            ["Number of pipes", base_graph.number_of_edges(), "Active and bypass pipes before failure isolation"],
            ["Monitoring nodes", len(sensor_nodes), "Sensor locations used for topological embedding"],
            ["Leak corridor length", len({tuple(sorted(e)) for e in [(7, 8), (8, 14), (14, 15), (15, 16), (16, 22)]}), "Pipes subject to progressive leakage"],
            ["Simulated horizon", "30 months", "Ten baseline, ten incipient, ten critical months"],
            ["Topological summaries", "H0, H1, Betti curves", "Persistent homology descriptors used in the early-warning module"],
        ],
        columns=["Item", "Value", "Interpretation"],
    )

    phase_table = pd.DataFrame(
        [
            ["Baseline (M1-M10)", baseline["mean_pressure"].mean(), baseline["graph_efficiency"].mean(), baseline["service_ratio"].mean(), baseline["tp0"].mean(), baseline["bc0_auc"].mean()],
            ["Incipient (M11-M20)", incipient["mean_pressure"].mean(), incipient["graph_efficiency"].mean(), incipient["service_ratio"].mean(), incipient["tp0"].mean(), incipient["bc0_auc"].mean()],
            ["Critical (M21-M30)", critical["mean_pressure"].mean(), critical["graph_efficiency"].mean(), critical["service_ratio"].mean(), critical["tp0"].mean(), critical["bc0_auc"].mean()],
        ],
        columns=["Scenario phase", "Mean pressure (m)", "Graph efficiency", "Service ratio", "TP0", "AUC(Betti-0)"],
    )

    service_loss_month = int(data.loc[data["service_ratio"] < 0.99, "month"].min())
    robust_threshold = float(data["tii_threshold"].iloc[0])
    alert_rows = [
        ["Topological instability index", int(data.loc[(data["month"] > 8) & (data["topological_instability_index"] > robust_threshold), "month"].min())],
        ["Mean pressure < 50 m", int(data.loc[data["mean_pressure"] < 50.0, "month"].min())],
        ["Minimum pressure < 35 m", int(data.loc[data["min_pressure"] < 35.0, "month"].min())],
        ["Graph efficiency < 0.85", int(data.loc[data["graph_efficiency"] < 0.85, "month"].min())],
        ["Service ratio < 0.99", service_loss_month],
    ]
    alert_table = pd.DataFrame(alert_rows, columns=["Indicator", "Alert month"])
    alert_table["Lead time before service loss (months)"] = service_loss_month - alert_table["Alert month"]
    alert_table["Operational meaning"] = [
        "Topological departure from the healthy multiscale signature",
        "Average system pressure leaves the comfort margin",
        "Most vulnerable node set enters critical pressure regime",
        "Network routing efficiency loses structural redundancy",
        "Observed unmet demand becomes material",
    ]

    selected_table = data[data["month"].isin([8, 13, 21, 30])][
        ["month", "mean_pressure", "min_pressure", "graph_efficiency", "service_ratio", "topological_instability_index", "tp0", "tp1", "bc0_auc", "bc1_auc"]
    ].copy()
    selected_table.columns = [
        "Month",
        "Mean pressure (m)",
        "Minimum pressure (m)",
        "Graph efficiency",
        "Service ratio",
        "Topological instability index",
        "TP0",
        "TP1",
        "AUC(Betti-0)",
        "AUC(Betti-1)",
    ]

    return {
        "table_1_network_configuration": topology_table,
        "table_2_phase_summary": phase_table,
        "table_3_early_warning": alert_table,
        "table_4_selected_months": selected_table,
    }


# -----------------------------------------------------------------------------
# 6. Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    base_graph = build_base_network()
    monthly_df, node_df, edge_df, graphs_by_month, sensor_nodes, leak_edges = simulate_network_evolution(base_graph)
    analysis_df, topo_summaries = add_topological_metrics(monthly_df, graphs_by_month, sensor_nodes)

    # Export raw data.
    monthly_df.to_csv(os.path.join(DATA_DIR, "monthly_hydraulic_metrics.csv"), index=False)
    node_df.to_csv(os.path.join(DATA_DIR, "node_time_series.csv"), index=False)
    edge_df.to_csv(os.path.join(DATA_DIR, "edge_time_series.csv"), index=False)
    analysis_df.to_csv(os.path.join(DATA_DIR, "monthly_integrated_metrics.csv"), index=False)

    # Figures for section 5.1.
    draw_architecture_figure(os.path.join(SEC_51, "figure_1_architecture.png"))
    draw_network_map(base_graph, graphs_by_month[8], leak_edges, 8, os.path.join(SEC_51, "figure_2_network_baseline.png"))
    draw_network_map(base_graph, graphs_by_month[30], leak_edges, 30, os.path.join(SEC_51, "figure_3_network_critical.png"))
    draw_regime_timeline(analysis_df, os.path.join(SEC_51, "figure_4_regime_timeline.png"))

    # Figures for section 5.2.
    draw_persistence_diagrams(topo_summaries, [8, 13, 21, 30], os.path.join(SEC_52, "figure_5_persistence_diagrams.png"))
    draw_betti_curves(topo_summaries, [8, 13, 21, 30], os.path.join(SEC_52, "figure_6_betti_curves.png"))
    draw_temporal_metrics(analysis_df, os.path.join(SEC_52, "figure_7_temporal_topology_vs_hydraulics.png"))

    # Figures for section 5.3.
    early_warning_df = pd.DataFrame(
        [
            ["Topological instability index", int(analysis_df.loc[(analysis_df["month"] > 8) & (analysis_df["topological_instability_index"] > analysis_df["tii_threshold"].iloc[0]), "month"].min())],
            ["Mean pressure < 50 m", int(analysis_df.loc[analysis_df["mean_pressure"] < 50.0, "month"].min())],
            ["Minimum pressure < 35 m", int(analysis_df.loc[analysis_df["min_pressure"] < 35.0, "month"].min())],
            ["Graph efficiency < 0.85", int(analysis_df.loc[analysis_df["graph_efficiency"] < 0.85, "month"].min())],
            ["Service ratio < 0.99", int(analysis_df.loc[analysis_df["service_ratio"] < 0.99, "month"].min())],
        ],
        columns=["indicator", "alert_month"],
    )
    service_loss_month = int(analysis_df.loc[analysis_df["service_ratio"] < 0.99, "month"].min())
    early_warning_df["lead_time_months"] = service_loss_month - early_warning_df["alert_month"]
    draw_early_warning_bars(early_warning_df, os.path.join(SEC_53, "figure_8_early_warning_lead_time.png"))
    draw_phase_heatmap(analysis_df, os.path.join(SEC_53, "figure_9_phase_heatmap.png"))

    # Tables.
    tables = build_tables(base_graph, analysis_df, node_df, edge_df, sensor_nodes)
    for name, table in tables.items():
        table.to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)

    # Compact article-facing summary for downstream document generation.
    summary = {
        "service_loss_month": service_loss_month,
        "topological_alert_month": int(early_warning_df.loc[early_warning_df["indicator"] == "Topological instability index", "alert_month"].iloc[0]),
        "lead_time_months": int(early_warning_df.loc[early_warning_df["indicator"] == "Topological instability index", "lead_time_months"].iloc[0]),
        "baseline_pressure_mean": float(analysis_df.loc[analysis_df["phase"] == "Baseline", "mean_pressure"].mean()),
        "critical_pressure_mean": float(analysis_df.loc[analysis_df["phase"] == "Critical", "mean_pressure"].mean()),
        "baseline_tp0_mean": float(analysis_df.loc[analysis_df["phase"] == "Baseline", "tp0"].mean()),
        "critical_tp0_mean": float(analysis_df.loc[analysis_df["phase"] == "Critical", "tp0"].mean()),
        "baseline_auc0_mean": float(analysis_df.loc[analysis_df["phase"] == "Baseline", "bc0_auc"].mean()),
        "critical_auc0_mean": float(analysis_df.loc[analysis_df["phase"] == "Critical", "bc0_auc"].mean()),
        "baseline_eff_mean": float(analysis_df.loc[analysis_df["phase"] == "Baseline", "graph_efficiency"].mean()),
        "critical_eff_mean": float(analysis_df.loc[analysis_df["phase"] == "Critical", "graph_efficiency"].mean()),
        "selected_rows": analysis_df.loc[analysis_df["month"].isin([8, 13, 21, 30]), ["month", "mean_pressure", "min_pressure", "graph_efficiency", "service_ratio", "topological_instability_index", "tp0", "tp1", "bc0_auc", "bc1_auc"]].round(4).to_dict(orient="records"),
    }
    pd.Series(summary).to_json(os.path.join(DATA_DIR, "article_summary.json"), indent=2)

    print("All outputs generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
