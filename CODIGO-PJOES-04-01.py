#!/usr/bin/env python3
"""Generate synthetic data, solve the Topological–Fuzzy Urban Water Allocation case study,
and export figures/tables used in the manuscript.

The script reproduces the computational experiment for:
Topological–Fuzzy Optimization of Urban Water Allocation Under Pollution and Sustainability Constraints
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

SOURCE_NAMES = ["Reservoir", "Aquifer", "Reclaimed", "Imported"]
SECTOR_NAMES = ["Residential", "Industrial", "Commercial", "Public services", "Urban green"]

COMPATIBILITY = np.array([
    [1.00, 1.00, 1.00, 1.00, 1.00],
    [0.65, 0.85, 0.70, 0.60, 0.80],
    [0.08, 0.60, 0.25, 0.18, 0.95],
    [1.00, 1.00, 1.00, 1.00, 1.00],
], dtype=float)

QUALITY_LIMITS = np.array([0.45, 0.78, 0.62, 0.50, 0.95], dtype=float)
MIN_COVERAGE = np.array([0.96, 0.88, 0.90, 0.95, 0.72], dtype=float)
PRIORITY_ORDER = [0, 3, 2, 1, 4]
NONPOTABLE_IDX = [1, 4]

CAP_FACTOR = np.array([
    [0.82, 0.82, 0.84, 0.88, 0.90, 0.88],
    [0.76, 0.74, 0.74, 0.78, 0.82, 0.80],
    [0.97, 0.97, 0.97, 0.97, 0.97, 0.97],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
], dtype=float)

A_LOW = np.array([0.84, 0.90, 0.94, 0.96])[:, None]
A_UP = np.array([1.05, 1.03, 1.02, 1.01])[:, None]
D_LOW = np.array([0.97, 0.96, 0.97, 0.98, 0.95])[:, None]
D_UP = np.array([1.09, 1.06, 1.07, 1.04, 1.12])[:, None]
R_LOW = np.array([0.95, 0.97, 0.96, 0.98])[:, None]
R_UP = np.array([1.16, 1.10, 1.12, 1.05])[:, None]
C_LOW = np.array([0.98, 0.97, 0.98, 0.98])[:, None]
C_UP = np.array([1.06, 1.09, 1.07, 1.11])[:, None]


def generate_synthetic_system(seed=42):
    rng = np.random.default_rng(seed)
    months = np.arange(1, 13)
    rain = 72 + 42 * np.sin(2 * np.pi * (months - 4) / 12) + rng.gamma(2.0, 6.5, 12) - 13
    rain = np.clip(rain, 22, None)
    temp = 22.8 + 4.6 * np.sin(2 * np.pi * (months - 2) / 12) + rng.normal(0, 0.65, 12)
    lag_rain = np.r_[rain[0], rain[:-1]]

    residential = 9300 + 105 * (temp - 22.5) - 7.5 * (rain - 70) + rng.normal(0, 85, 12)
    industrial = 4760 + 18 * (temp - 22.5) + 0.20 * (rain - 70) + rng.normal(0, 55, 12)
    commercial = 3200 + 16 * (temp - 22.5) - 0.15 * (rain - 70) + rng.normal(0, 40, 12)
    public = 1900 + 8 * (temp - 22.5) + rng.normal(0, 22, 12)
    green = 1700 + 85 * (temp - 22.5) - 8.5 * (rain - 70) + rng.normal(0, 60, 12)
    demands = np.vstack([residential, industrial, commercial, public, green]).T
    demands = np.clip(demands, [8500, 4300, 2800, 1650, 900], None)
    wastewater_return = 0.58 * residential + 0.28 * industrial + 0.22 * commercial + 0.85 * public

    reservoir = 7600 + 11.0 * rain + 18 * np.maximum(rain - 90, 0) - 55 * np.maximum(temp - 26, 0) + rng.normal(0, 120, 12)
    aquifer = 6480 + 1.8 * lag_rain - 16 * np.maximum(temp - 25, 0) + rng.normal(0, 90, 12)
    reclaimed = 0.94 * 0.20 * wastewater_return + 300 + 0.4 * np.maximum(temp - 24, 0) + rng.normal(0, 80, 12)
    imported = 2280 + 1.2 * np.maximum(75 - rain, 0) + rng.normal(0, 60, 12)
    avail = np.vstack([reservoir, aquifer, reclaimed, imported]).T
    avail = np.clip(avail, [5400, 5200, 4300, 1800], None)

    turbidity_shock = np.maximum(rain - 95, 0) / 25
    reservoir_risk = 0.42 + 0.09 * np.maximum(rain - 85, 0) / 20 + 0.02 * np.maximum(temp - 25, 0) + rng.normal(0, 0.025, 12)
    aquifer_risk = 0.28 + 0.02 * np.maximum(temp - 24, 0) + 0.015 * np.maximum(70 - rain, 0) / 20 + rng.normal(0, 0.012, 12)
    reclaimed_risk = 0.50 + 0.015 * np.maximum(temp - 25, 0) + 0.010 * np.maximum(wastewater_return - np.median(wastewater_return), 0) / 600 + rng.normal(0, 0.018, 12)
    imported_risk = 0.14 + 0.008 * np.maximum(rain - 95, 0) / 20 + rng.normal(0, 0.008, 12)
    risk = np.vstack([reservoir_risk, aquifer_risk, reclaimed_risk, imported_risk]).T
    risk = np.clip(risk, [0.18, 0.20, 0.34, 0.08], [1.25, 0.65, 0.95, 0.28])

    energy_index = 1 + 0.025 * np.maximum(temp - 24, 0)
    reservoir_cost = 121 + 3.8 * turbidity_shock + rng.normal(0, 1.2, 12)
    aquifer_cost = 163 * energy_index + rng.normal(0, 1.5, 12)
    reclaimed_cost = 178 + 2.1 * np.maximum(temp - 24, 0) + 3.0 * np.maximum(risk[:, 2] - 0.5, 0) * 10 + rng.normal(0, 1.5, 12)
    imported_cost = 240 + 1.8 * np.maximum(75 - rain, 0) / 10 + rng.normal(0, 2.0, 12)
    cost = np.vstack([reservoir_cost, aquifer_cost, reclaimed_cost, imported_cost]).T

    return dict(months=months, rain=rain, temp=temp, demands_m=demands, avail_m=avail, risk_m=risk, cost_m=cost)


def aggregate_bimonthly(arr, mode="sum"):
    arr = np.asarray(arr)
    chunks = []
    for k in range(0, 12, 2):
        block = arr[k:k + 2]
        chunks.append(block.sum(axis=0) if mode == "sum" else block.mean(axis=0))
    return np.array(chunks)


def defuzzify_tri(l, m, u):
    return (l + m + u) / 3.0


def apply_scenario(A, D, R, C, scenario):
    A = A.copy()
    D = D.copy()
    R = R.copy()
    C = C.copy()
    if scenario == "baseline":
        pass
    elif scenario == "pollution":
        R[0, 2:4] *= 1.28
        R[2, 2:4] *= 1.15
        C[0, 2:4] *= 1.06
        C[2, 2:4] *= 1.05
        A[0, 2:4] *= 0.96
    elif scenario == "compound":
        A[0, 1:5] *= np.array([0.96, 0.90, 0.86, 0.88])
        A[1, 1:5] *= np.array([0.95, 0.92, 0.90, 0.92])
        A[3, 2:5] *= 1.08
        D[:, 2:5] *= np.array([1.02, 1.04, 1.03])
        D[0, 2:5] *= 1.03
        D[4, 2:5] *= 1.07
        R[0, 2:5] *= np.array([1.18, 1.24, 1.15])
        R[2, 2:5] *= np.array([1.08, 1.10, 1.07])
        C[3, 2:5] *= 1.04
        C[2, 2:5] *= 1.03
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return A, D, R, C


def build_case_data(scenario="compound", use_fuzzy=True, seed=42):
    syn = generate_synthetic_system(seed)
    A = aggregate_bimonthly(syn["avail_m"], "sum").T
    D = aggregate_bimonthly(syn["demands_m"], "sum").T
    R = aggregate_bimonthly(syn["risk_m"], "mean").T
    C = aggregate_bimonthly(syn["cost_m"], "mean").T
    A, D, R, C = apply_scenario(A, D, R, C, scenario)
    if scenario == "compound":
        A[2, 1:4] *= 1.05
        A[3, :] *= 1.08
    if use_fuzzy:
        A = defuzzify_tri(A_LOW * A, A, A_UP * A)
        D = defuzzify_tri(D_LOW * D, D, D_UP * D)
        R = defuzzify_tri(R_LOW * R, R, np.minimum(R_UP * R, 1.60))
        C = defuzzify_tri(C_LOW * C, C, C_UP * C)
    L = A * CAP_FACTOR
    budget = D.sum(axis=0) * 165.0
    annual_gw_cap = 0.92 * A[1].sum()
    return {
        "A": A, "D": D, "R": R, "C": C, "L": L, "budget": budget, "annual_gw_cap": annual_gw_cap,
        "quality_limits": QUALITY_LIMITS, "min_coverage": MIN_COVERAGE, "compatibility": COMPATIBILITY,
        "nonpotable_idx": NONPOTABLE_IDX, "priority_order": PRIORITY_ORDER, "reuse_min_share": 0.18,
        "source_names": SOURCE_NAMES, "sector_names": SECTOR_NAMES, "synthetic": syn, "scenario": scenario, "seed": seed
    }


def decode_preferences(z, data):
    A, D, U = data["A"], data["D"], data["compatibility"]
    S, T = A.shape
    J = D.shape[0]
    pref = np.asarray(z, dtype=float).reshape(S, J, T)
    pref = np.clip(pref, 1e-6, None)
    x = np.zeros((S, J, T), dtype=float)
    for t in range(T):
        for j in range(J):
            w = pref[:, j, t] * U[:, j] + 1e-9
            x[:, j, t] = D[j, t] * w / w.sum()
            max_alloc = U[:, j] * D[j, t]
            x[:, j, t] = np.minimum(x[:, j, t], max_alloc)
            if x[:, j, t].sum() < D[j, t]:
                rem = D[j, t] - x[:, j, t].sum()
                resid = np.maximum(0, max_alloc - x[:, j, t])
                if resid.sum() > 0:
                    x[:, j, t] += rem * resid / resid.sum()
        src_use = x[:, :, t].sum(axis=1)
        for s in range(S):
            if src_use[s] > A[s, t]:
                x[s, :, t] *= A[s, t] / src_use[s]
        for _ in range(3):
            deficits = np.maximum(0, D[:, t] - x[:, :, t].sum(axis=0))
            slacks = np.maximum(0, A[:, t] - x[:, :, t].sum(axis=1))
            if deficits.sum() < 1e-8 or slacks.sum() < 1e-8:
                break
            for j in data["priority_order"]:
                deficit = deficits[j]
                if deficit <= 1e-8:
                    continue
                desir = pref[:, j, t] * U[:, j]
                capacity = np.minimum(slacks, U[:, j] * D[j, t] - x[:, j, t])
                capacity = np.maximum(0, capacity)
                if capacity.sum() <= 1e-8:
                    continue
                w = desir * capacity + 1e-12
                add = deficit * w / w.sum()
                add = np.minimum(add, capacity)
                if add.sum() < deficit and capacity.sum() > add.sum():
                    rem = deficit - add.sum()
                    resid = np.maximum(0, capacity - add)
                    if resid.sum() > 0:
                        add += rem * resid / resid.sum()
                x[:, j, t] += add
                slacks -= add
        for s in range(S):
            su = x[s, :, t].sum()
            if su > A[s, t]:
                x[s, :, t] *= A[s, t] / su
        for j in range(J):
            dj = x[:, j, t].sum()
            if dj > D[j, t]:
                x[:, j, t] *= D[j, t] / dj
    return x


def evaluate_primary(x, data):
    A, D, R, C, L = data["A"], data["D"], data["R"], data["C"], data["L"]
    S, J, T = x.shape
    direct_cost = float((x * C[:, None, :]).sum())
    max_cost = float((D.sum(axis=0) * C.max(axis=0)).sum())
    cost_norm = direct_cost / max_cost

    supplied = x.sum(axis=0)
    coverage = supplied / np.maximum(D, 1e-9)
    shortage = np.maximum(0, 1 - coverage)
    shares = x / np.maximum(supplied[None, :, :], 1e-9)
    hhi = (shares ** 2).sum(axis=0)
    stress = x.sum(axis=1) / np.maximum(A, 1e-9)
    vulnerability = 0.56 * shortage.mean() + 0.24 * ((hhi - 1 / S) / (1 - 1 / S)).mean() + 0.20 * np.minimum(stress, 1.5).mean()

    delivered_risk = (x * R[:, None, :]).sum(axis=0) / np.maximum(supplied, 1e-9)
    risk_ratio = delivered_risk / data["quality_limits"][:, None]
    pressure = x.sum(axis=1) / np.maximum(L, 1e-9)
    environment = 0.58 * np.average(np.minimum(risk_ratio, 2.0), weights=np.maximum(supplied, 1e-9)) + 0.42 * np.mean(np.minimum(pressure, 2.0))
    environment /= 2.0

    penalty = 0.0
    vio = {}

    env_ex = np.maximum(0, x.sum(axis=1) - L)
    vio["env_cap"] = float(env_ex.sum() / np.maximum(L.sum(), 1e-9))
    penalty += 10.0 * float(np.mean((env_ex / np.maximum(L, 1e-9)) ** 2))

    period_cost = (x * C[:, None, :]).sum(axis=(0, 1))
    budget_ex = np.maximum(0, period_cost - data["budget"])
    vio["budget"] = float(budget_ex.sum() / np.maximum(data["budget"].sum(), 1e-9))
    penalty += 8.0 * float(np.mean((budget_ex / np.maximum(data["budget"], 1e-9)) ** 2))

    cov_floor_ex = np.maximum(0, data["min_coverage"][:, None] - coverage)
    vio["coverage_floor"] = float(cov_floor_ex.mean())
    penalty += 14.0 * float(np.mean(cov_floor_ex ** 2))

    qual_ex = np.maximum(0, delivered_risk - data["quality_limits"][:, None])
    vio["quality"] = float((qual_ex / np.maximum(data["quality_limits"][:, None], 1e-9)).mean())
    penalty += 11.0 * float(np.mean((qual_ex / np.maximum(data["quality_limits"][:, None], 1e-9)) ** 2))

    gw_use = float(x[1].sum())
    gw_ex = max(0.0, gw_use - float(data["annual_gw_cap"]))
    vio["gw_annual"] = gw_ex / max(float(data["annual_gw_cap"]), 1e-9)
    penalty += 6.0 * (gw_ex / max(float(data["annual_gw_cap"]), 1e-9)) ** 2

    np_served = float(x[:, data["nonpotable_idx"], :].sum())
    reuse_share = float(x[2, data["nonpotable_idx"], :].sum() / max(np_served, 1e-9))
    reuse_ex = max(0.0, data["reuse_min_share"] - reuse_share)
    vio["reuse_min_share"] = reuse_ex / max(float(data["reuse_min_share"]), 1e-9)
    penalty += 4.0 * (reuse_ex / max(float(data["reuse_min_share"]), 1e-9)) ** 2

    objective = 0.36 * cost_norm + 0.34 * vulnerability + 0.30 * environment + penalty

    return {
        "objective": float(objective), "direct_cost": direct_cost, "cost_norm": float(cost_norm),
        "vulnerability": float(vulnerability), "environment": float(environment), "penalty": float(penalty),
        "coverage_mean": float(coverage.mean()), "shortage_ratio": float(shortage.mean()), "hhi_mean": float(hhi.mean()),
        "pressure_mean": float(pressure.mean()), "quality_ratio_mean": float(risk_ratio.mean()), "reuse_share": reuse_share,
        "gw_use": gw_use, "period_cost": period_cost, "coverage": coverage, "delivered_risk": delivered_risk,
        "violations": vio, "x": x
    }


def evaluate_preferences(z, data):
    x = decode_preferences(z, data)
    res = evaluate_primary(x, data)
    res["z"] = np.asarray(z).copy()
    return res


def pairwise_distance(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def vr_persistence_h0_h1(dist):
    n = dist.shape[0]
    simplices = []
    for i in range(n):
        simplices.append((0.0, (i,), 0))
    for i in range(n):
        for j in range(i + 1, n):
            simplices.append((float(dist[i, j]), (i, j), 1))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                simplices.append((float(max(dist[i, j], dist[i, k], dist[j, k])), (i, j, k), 2))
    simplices.sort(key=lambda x: (x[0], x[2], x[1]))
    index = {s[1]: idx for idx, s in enumerate(simplices)}
    boundaries = []
    for _, verts, dim in simplices:
        if dim == 0:
            boundaries.append([])
        elif dim == 1:
            boundaries.append(sorted([index[(verts[0],)], index[(verts[1],)]]))
        else:
            faces = [(verts[0], verts[1]), (verts[0], verts[2]), (verts[1], verts[2])]
            boundaries.append(sorted([index[f] for f in faces]))
    reduced = {}
    pivot_to_col = {}
    pairs = []
    unpaired = set(range(len(simplices)))
    for j in range(len(simplices)):
        col = boundaries[j][:]
        while col and col[-1] in pivot_to_col:
            other = reduced[pivot_to_col[col[-1]]]
            col = sorted(set(col).symmetric_difference(other))
        reduced[j] = col
        if col:
            p = col[-1]
            pivot_to_col[p] = j
            pairs.append((p, j))
            unpaired.discard(p)
            unpaired.discard(j)
    intervals = {0: [], 1: [], 2: []}
    essential = {0: [], 1: [], 2: []}
    for b, d in pairs:
        dim = simplices[b][2]
        intervals[dim].append((simplices[b][0], simplices[d][0]))
    for idx in sorted(unpaired):
        dim = simplices[idx][2]
        essential[dim].append((simplices[idx][0], math.inf))
    return simplices, intervals, essential


def scenario_topology18(x, data):
    groups = [("Potable core", [0, 3]), ("Economic", [1, 2]), ("Urban green", [4])]
    _, _, T = x.shape
    pts = []
    for t in range(T):
        supplied_j = x[:, :, t].sum(axis=0)
        delivered_risk_j = (x[:, :, t] * data["R"][:, t][:, None]).sum(axis=0) / np.maximum(supplied_j, 1e-9)
        coverage_j = supplied_j / np.maximum(data["D"][:, t], 1e-9)
        for gi, (_, idx) in enumerate(groups):
            xg = x[:, idx, t].sum(axis=1)
            supplied = xg.sum()
            shares = xg / max(supplied, 1e-9)
            coverage = np.average(coverage_j[idx], weights=np.maximum(data["D"][idx, t], 1e-9))
            risk_slack = 1 - np.average(delivered_risk_j[idx] / data["quality_limits"][idx], weights=np.maximum(supplied_j[idx], 1e-9))
            demand_share = data["D"][idx, t].sum() / max(data["D"][:, t].sum(), 1e-9)
            pts.append(np.r_[shares, coverage, risk_slack, demand_share, t / max(T - 1, 1), gi / 2.0])
    pts = np.array(pts)
    Dm = pairwise_distance(pts)
    if Dm.max() > 0:
        Dm = Dm / Dm.max()
    _, intervals, essential = vr_persistence_h0_h1(Dm)
    h0 = [(b, d) for b, d in intervals[0] if math.isfinite(d) and d > b + 1e-9]
    h1 = [(b, d) for b, d in intervals[1] if math.isfinite(d) and d > b + 1e-9]
    p0 = float(sum(d - b for b, d in h0))
    p1 = float(sum(d - b for b, d in h1))
    return {"dist": Dm, "h0": h0, "h1": h1, "p0": p0, "p1": p1, "pts": pts, "essential": essential}


def topo_fragility_from_topo(topo):
    return float(100.0 * topo["p1"] / (topo["p0"] + 1e-9))


def persistence_landscape(intervals, x_grid=None, k_max=3):
    if x_grid is None:
        max_d = max([d for _, d in intervals], default=1.0)
        x_grid = np.linspace(0, max(max_d * 1.05, 1e-6), 300)
    tents = []
    for b, d in intervals:
        vals = np.maximum(0, np.minimum(x_grid - b, d - x_grid))
        tents.append(vals)
    if len(tents) == 0:
        return x_grid, np.zeros((k_max, len(x_grid)))
    T = np.vstack(tents)
    lambdas = np.zeros((k_max, len(x_grid)))
    for i in range(len(x_grid)):
        col = np.sort(T[:, i])[::-1]
        for k in range(min(k_max, len(col))):
            lambdas[k, i] = col[k]
    return x_grid, lambdas


def landscape_area(intervals, k_max=3):
    xg, lams = persistence_landscape(intervals, None, k_max)
    return float(np.trapezoid(lams.sum(axis=0), xg))


@dataclass
class Evaluator:
    data: dict
    evals: int = 0
    best: dict = None
    history: list = field(default_factory=list)
    archive: list = field(default_factory=list)

    def evaluate(self, z):
        self.evals += 1
        res = evaluate_preferences(z, self.data)
        self.archive.append(res)
        if self.best is None or res["objective"] < self.best["objective"]:
            self.best = res
        self.history.append((self.evals, float(self.best["objective"])))
        return res


def heuristic_candidate(data, mode="balanced", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    A, D, R, C = data["A"], data["D"], data["R"], data["C"]
    S, T = A.shape
    J = D.shape[0]
    z = np.zeros((S, J, T), dtype=float)
    for t in range(T):
        for j in range(J):
            if mode == "cost":
                score = 1.0 / (C[:, t] + 1e-6)
            elif mode == "risk":
                score = np.clip(1.05 - R[:, t], 0.05, None)
            elif mode == "reuse":
                score = np.clip(1.0 - R[:, t], 0.05, None) / (C[:, t] + 1e-6)
                if j in data["nonpotable_idx"]:
                    score[2] *= 1.5
            else:
                score = np.clip(1.05 - R[:, t], 0.05, None) / (C[:, t] + 1e-6)
                score *= (A[:, t] / A[:, t].max()) ** 0.4
                if j in data["nonpotable_idx"]:
                    score[2] *= 1.35
                if j in [0, 3]:
                    score[3] *= 1.15
                    score[1] *= 1.10
            score = score * data["compatibility"][:, j]
            z[:, j, t] = np.maximum(1e-4, score * (1 + 0.08 * rng.normal(size=S)))
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return z.ravel()


def run_greedy(data, seed=1):
    rng = np.random.default_rng(seed)
    ev = Evaluator(data)
    for mode in ["balanced", "cost", "risk", "reuse"]:
        for _ in range(8):
            z = heuristic_candidate(data, mode, rng)
            z = np.clip(z + rng.normal(0, 0.04, size=z.size), 0, 1)
            ev.evaluate(z)
    return ev


def run_de(data, seed=1, budget=600, pop_size=24):
    rng = np.random.default_rng(seed)
    d = data["A"].shape[0] * data["D"].shape[0] * data["A"].shape[1]
    ev = Evaluator(data)
    pop = []
    fit = []
    heur_modes = ["balanced", "cost", "risk", "reuse"]
    for i in range(pop_size):
        if i < len(heur_modes):
            z = heuristic_candidate(data, heur_modes[i], rng)
            z = np.clip(z + rng.normal(0, 0.05, size=d), 0, 1)
        else:
            z = rng.random(d)
        res = ev.evaluate(z)
        pop.append(z)
        fit.append(res["objective"])
        if ev.evals >= budget:
            break
    pop = np.array(pop)
    fit = np.array(fit)
    F = 0.65
    CR = 0.85
    gen = 0
    while ev.evals < budget:
        for i in range(len(pop)):
            if ev.evals >= budget:
                break
            ids = [idx for idx in range(len(pop)) if idx != i]
            a, b, c = pop[rng.choice(ids, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), 0, 1)
            mask = rng.random(d) < CR
            mask[rng.integers(d)] = True
            trial = np.where(mask, mutant, pop[i])
            res = ev.evaluate(trial)
            if res["objective"] < fit[i]:
                pop[i] = trial
                fit[i] = res["objective"]
        gen += 1
        if gen % 10 == 0:
            F = 0.55 + 0.25 * rng.random()
            CR = 0.75 + 0.20 * rng.random()
    return ev


def run_ga(data, seed=1, budget=600, pop_size=24):
    rng = np.random.default_rng(seed)
    d = data["A"].shape[0] * data["D"].shape[0] * data["A"].shape[1]
    ev = Evaluator(data)
    pop = []
    fit = []
    heur_modes = ["balanced", "cost", "risk", "reuse"]
    for i in range(pop_size):
        if i < len(heur_modes):
            z = heuristic_candidate(data, heur_modes[i], rng)
            z = np.clip(z + rng.normal(0, 0.05, size=d), 0, 1)
        else:
            z = rng.random(d)
        res = ev.evaluate(z)
        pop.append(z)
        fit.append(res["objective"])
        if ev.evals >= budget:
            break
    pop = np.array(pop)
    fit = np.array(fit)
    while ev.evals < budget:
        ranks = np.argsort(np.argsort(fit))
        sel_prob = (len(fit) - ranks) / (len(fit) * (len(fit) + 1) / 2)
        new_pop = []
        elite_idx = np.argsort(fit)[:2]
        new_pop.extend(pop[elite_idx])
        while len(new_pop) < len(pop) and ev.evals < budget:
            p1, p2 = pop[rng.choice(len(pop), 2, replace=False, p=sel_prob)]
            cut = rng.integers(1, d - 1)
            child = np.r_[p1[:cut], p2[cut:]]
            mut_mask = rng.random(d) < 0.08
            child[mut_mask] = np.clip(child[mut_mask] + rng.normal(0, 0.18, mut_mask.sum()), 0, 1)
            if rng.random() < 0.25:
                child = 0.85 * child + 0.15 * heuristic_candidate(data, "balanced", rng)
                child = np.clip(child, 0, 1)
            new_pop.append(child)
        pop = np.array(new_pop[:len(pop)])
        fit = np.array([ev.evaluate(ind)["objective"] for ind in pop])
    return ev


def local_improve(z, data, rng, ev, step=0.12, tries=2):
    best_z = z.copy()
    best = ev.evaluate(best_z)
    for _ in range(tries):
        cand = np.clip(best_z + rng.normal(0, step, size=best_z.size), 0, 1)
        res = ev.evaluate(cand)
        if res["objective"] < best["objective"]:
            best = res
            best_z = cand
            step *= 0.85
        else:
            step *= 0.92
    return best_z, best


def run_fho(data, seed=1, budget=600, pop_size=24):
    rng = np.random.default_rng(seed)
    d = data["A"].shape[0] * data["D"].shape[0] * data["A"].shape[1]
    ev = Evaluator(data)
    pop = []
    fit = []
    modes = ["balanced", "cost", "risk", "reuse"]
    for i in range(pop_size):
        mode = modes[i % len(modes)]
        base = heuristic_candidate(data, mode, rng)
        noise = rng.normal(0, 0.07 if i < len(modes) else 0.18, size=d)
        z = np.clip(base + noise, 0, 1) if i < pop_size // 2 else rng.random(d)
        res = ev.evaluate(z)
        pop.append(z)
        fit.append(res["objective"])
        if ev.evals >= budget:
            break
    pop = np.array(pop)
    fit = np.array(fit)
    archive = []
    gen = 0
    while ev.evals < budget:
        order = np.argsort(fit)
        pop = pop[order]
        fit = fit[order]
        archive.extend([ev.archive[idx] for idx in range(max(0, len(ev.archive) - len(pop)), len(ev.archive))])
        archive = sorted(archive, key=lambda r: r["objective"])[:40]
        elite = pop[:max(4, len(pop) // 4)]
        new_pop = [elite[0].copy(), elite[1].copy()]
        while len(new_pop) < len(pop) and ev.evals < budget:
            u = rng.random()
            if u < 0.40:
                ids = rng.choice(len(elite), 3, replace=False)
                a, b, c = elite[ids]
                cand = np.clip(a + 0.62 * (b - c) + rng.normal(0, 0.03, size=d), 0, 1)
            elif u < 0.75:
                p1, p2 = elite[rng.choice(len(elite), 2, replace=False)]
                alpha = rng.random()
                cand = np.clip(alpha * p1 + (1 - alpha) * p2 + rng.normal(0, 0.04, size=d), 0, 1)
            else:
                cand = np.clip(0.75 * heuristic_candidate(data, "balanced", rng) + 0.25 * rng.random(d), 0, 1)
            if rng.random() < 0.30:
                cand = np.clip(cand + rng.normal(0, 0.12, size=d), 0, 1)
            new_pop.append(cand)
        pop = np.array(new_pop[:len(pop)])
        fit = np.zeros(len(pop))
        for i in range(len(pop)):
            if ev.evals >= budget:
                break
            res = ev.evaluate(pop[i])
            fit[i] = res["objective"]
        gen += 1
        if gen % 8 == 0:
            top_ids = np.argsort(fit)[:2]
            for idx in top_ids:
                if ev.evals >= budget:
                    break
                z_new, res_new = local_improve(pop[idx], data, rng, ev, step=0.10, tries=2)
                pop[idx] = z_new
                fit[idx] = res_new["objective"]
    archive = sorted(ev.archive, key=lambda r: r["objective"])[:80]
    ev.elite_archive = archive
    return ev


def select_topological_elite(archive, data, tol=1.03):
    best_obj = min(r["objective"] for r in archive)
    cand = [r for r in archive if r["objective"] <= best_obj * tol]
    best = None
    for r in cand:
        topo = scenario_topology18(r["x"], data)
        frag = topo_fragility_from_topo(topo)
        area = landscape_area(topo["h1"], 3)
        if best is None or (frag, area, r["objective"]) < (best["topo_fragility"], best["landscape_area"], best["objective"]):
            rr = dict(r)
            rr["topo"] = topo
            rr["topo_fragility"] = frag
            rr["landscape_area"] = area
            best = rr
    return best


def run_fho_with_tfho(data, seed=1, budget=600, pop_size=24, tol=1.03):
    ev = run_fho(data, seed, budget, pop_size)
    tfho = select_topological_elite(ev.elite_archive, data, tol)
    return ev, tfho


def run_all_methods(data, seeds=range(1, 6), budget=600, pop_size=24):
    rows = []
    histories = {}
    reps = {}
    for seed in seeds:
        t0 = time.time()
        ev = run_greedy(data, seed=seed)
        dt = time.time() - t0
        b = ev.best
        rows.append({"method": "Greedy", "seed": seed, "objective": b["objective"], "direct_cost": b["direct_cost"],
                     "cost_norm": b["cost_norm"], "vulnerability": b["vulnerability"], "environment": b["environment"],
                     "penalty": b["penalty"], "coverage": b["coverage_mean"], "shortage": b["shortage_ratio"],
                     "reuse": b["reuse_share"], "fragility": topo_fragility_from_topo(scenario_topology18(b["x"], data)),
                     "runtime": dt})
        histories.setdefault("Greedy", []).append(ev.history)
        reps[("Greedy", seed)] = b

        t0 = time.time()
        ev = run_de(data, seed=seed, budget=budget, pop_size=pop_size)
        dt = time.time() - t0
        b = ev.best
        rows.append({"method": "DE", "seed": seed, "objective": b["objective"], "direct_cost": b["direct_cost"],
                     "cost_norm": b["cost_norm"], "vulnerability": b["vulnerability"], "environment": b["environment"],
                     "penalty": b["penalty"], "coverage": b["coverage_mean"], "shortage": b["shortage_ratio"],
                     "reuse": b["reuse_share"], "fragility": topo_fragility_from_topo(scenario_topology18(b["x"], data)),
                     "runtime": dt})
        histories.setdefault("DE", []).append(ev.history)
        reps[("DE", seed)] = b

        t0 = time.time()
        ev = run_ga(data, seed=seed, budget=budget, pop_size=pop_size)
        dt = time.time() - t0
        b = ev.best
        rows.append({"method": "GA", "seed": seed, "objective": b["objective"], "direct_cost": b["direct_cost"],
                     "cost_norm": b["cost_norm"], "vulnerability": b["vulnerability"], "environment": b["environment"],
                     "penalty": b["penalty"], "coverage": b["coverage_mean"], "shortage": b["shortage_ratio"],
                     "reuse": b["reuse_share"], "fragility": topo_fragility_from_topo(scenario_topology18(b["x"], data)),
                     "runtime": dt})
        histories.setdefault("GA", []).append(ev.history)
        reps[("GA", seed)] = b

        t0 = time.time()
        ev, tfho = run_fho_with_tfho(data, seed=seed, budget=budget, pop_size=pop_size, tol=1.03)
        dt = time.time() - t0
        b = ev.best
        rows.append({"method": "FHO", "seed": seed, "objective": b["objective"], "direct_cost": b["direct_cost"],
                     "cost_norm": b["cost_norm"], "vulnerability": b["vulnerability"], "environment": b["environment"],
                     "penalty": b["penalty"], "coverage": b["coverage_mean"], "shortage": b["shortage_ratio"],
                     "reuse": b["reuse_share"], "fragility": topo_fragility_from_topo(scenario_topology18(b["x"], data)),
                     "runtime": dt})
        rows.append({"method": "TFHO", "seed": seed, "objective": tfho["objective"], "direct_cost": tfho["direct_cost"],
                     "cost_norm": tfho["cost_norm"], "vulnerability": tfho["vulnerability"], "environment": tfho["environment"],
                     "penalty": tfho["penalty"], "coverage": tfho["coverage_mean"], "shortage": tfho["shortage_ratio"],
                     "reuse": tfho["reuse_share"], "fragility": tfho["topo_fragility"], "runtime": dt})
        histories.setdefault("FHO", []).append(ev.history)
        histories.setdefault("TFHO", []).append(ev.history)
        reps[("FHO", seed)] = b
        reps[("TFHO", seed)] = tfho
    return pd.DataFrame(rows), histories, reps


def benchmark_scenarios_fho_tfho(seed_syn=42, scenarios=("baseline", "pollution", "compound"), seeds=range(1, 6), budget=600, pop_size=24):
    rows = []
    reps_by_scenario = {}
    for sc in scenarios:
        data = build_case_data(sc, True, seed=seed_syn)
        rep_candidates = {}
        fho_rows = []
        for seed in seeds:
            ev, tfho = run_fho_with_tfho(data, seed=seed, budget=budget, pop_size=pop_size, tol=1.03)
            fho = ev.best
            frag_fho = topo_fragility_from_topo(scenario_topology18(fho["x"], data))
            row_fho = {"scenario": sc, "method": "FHO", "seed": seed, "objective": fho["objective"], "direct_cost": fho["direct_cost"],
                       "cost_norm": fho["cost_norm"], "vulnerability": fho["vulnerability"], "environment": fho["environment"],
                       "penalty": fho["penalty"], "coverage": fho["coverage_mean"], "shortage": fho["shortage_ratio"],
                       "reuse": fho["reuse_share"], "fragility": frag_fho}
            row_tf = {"scenario": sc, "method": "TFHO", "seed": seed, "objective": tfho["objective"], "direct_cost": tfho["direct_cost"],
                      "cost_norm": tfho["cost_norm"], "vulnerability": tfho["vulnerability"], "environment": tfho["environment"],
                      "penalty": tfho["penalty"], "coverage": tfho["coverage_mean"], "shortage": tfho["shortage_ratio"],
                      "reuse": tfho["reuse_share"], "fragility": tfho["topo_fragility"]}
            rows.extend([row_fho, row_tf])
            rep_candidates[seed] = (fho, tfho, data)
            fho_rows.append(row_fho)
        dff = pd.DataFrame(fho_rows)
        med = dff.objective.median()
        seed_rep = int((dff.assign(dist=(dff.objective - med).abs()).sort_values("dist").iloc[0]["seed"]))
        reps_by_scenario[sc] = {"seed": seed_rep, "FHO": rep_candidates[seed_rep][0], "TFHO": rep_candidates[seed_rep][1], "data": data}
    return pd.DataFrame(rows), reps_by_scenario


def mean_sd(series):
    return f"{series.mean():.4f} ± {series.std():.4f}"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_outputs(outdir: Path, seed: int = 42, seeds=range(1, 6), budget: int = 600, pop_size: int = 24):
    outdir = ensure_dir(outdir)
    data_comp = build_case_data("compound", True, seed)
    bench_df, histories, reps = run_all_methods(data_comp, seeds=seeds, budget=budget, pop_size=pop_size)
    sc_df, scen_reps = benchmark_scenarios_fho_tfho(seed_syn=seed, scenarios=("baseline", "pollution", "compound"), seeds=seeds, budget=budget, pop_size=pop_size)

    baseline_modal = build_case_data("baseline", use_fuzzy=False, seed=seed)
    source_summary = pd.DataFrame({
        "Source": SOURCE_NAMES,
        "Mean availability (10^3 m3/period)": baseline_modal["A"].mean(axis=1),
        "Mean risk index": baseline_modal["R"].mean(axis=1),
        "Mean unit cost (USD/10^3 m3)": baseline_modal["C"].mean(axis=1),
        "Low fuzzy factor": A_LOW[:, 0],
        "High fuzzy factor": A_UP[:, 0],
    })
    sector_summary = pd.DataFrame({
        "Sector": SECTOR_NAMES,
        "Mean demand (10^3 m3/period)": baseline_modal["D"].mean(axis=1),
        "Quality limit": QUALITY_LIMITS,
        "Minimum coverage": MIN_COVERAGE,
        "Demand low factor": D_LOW[:, 0],
        "Demand high factor": D_UP[:, 0],
    })
    source_summary.round(3).to_csv(outdir / "table_1_sources.csv", index=False)
    sector_summary.round(3).to_csv(outdir / "table_2_sectors.csv", index=False)

    tab3 = bench_df.groupby("method").agg({
        "objective": mean_sd,
        "cost_norm": mean_sd,
        "vulnerability": mean_sd,
        "environment": mean_sd,
        "penalty": mean_sd,
        "coverage": mean_sd,
        "shortage": mean_sd,
        "reuse": mean_sd,
        "fragility": mean_sd,
        "runtime": mean_sd,
    }).reset_index()
    tab3.columns = ["Method", "Objective", "Cost norm", "Vulnerability", "Environment", "Penalty",
                    "Coverage", "Shortage", "Reuse share", "Topological fragility", "Runtime (s)"]
    tab3.to_csv(outdir / "table_3_compound_benchmark.csv", index=False)

    tab4 = sc_df.groupby(["scenario", "method"]).agg({"objective": "mean", "coverage": "mean", "reuse": "mean", "fragility": "mean"}).reset_index()
    tab4.to_csv(outdir / "table_4_scenario_sensitivity.csv", index=False)

    ev2, tf2 = run_fho_with_tfho(data_comp, seed=2, budget=budget, pop_size=pop_size, tol=1.03)
    alloc_fho = pd.DataFrame(np.round(ev2.best["x"].sum(axis=2), 1), index=SOURCE_NAMES, columns=SECTOR_NAMES)
    alloc_tf = pd.DataFrame(np.round(tf2["x"].sum(axis=2), 1), index=SOURCE_NAMES, columns=SECTOR_NAMES)
    alloc_fho.to_csv(outdir / "table_5a_compound_alloc_fho_seed2.csv")
    alloc_tf.to_csv(outdir / "table_5b_compound_alloc_tfho_seed2.csv")
    cov_df = pd.DataFrame({
        "Sector": SECTOR_NAMES,
        "FHO mean coverage": np.round(ev2.best["coverage"].mean(axis=1), 3),
        "TFHO mean coverage": np.round(tf2["coverage"].mean(axis=1), 3),
        "Quality limit": QUALITY_LIMITS,
        "Minimum coverage": MIN_COVERAGE,
    })
    cov_df.to_csv(outdir / "table_5c_compound_coverage_seed2.csv", index=False)

    bench_df.to_csv(outdir / "benchmark_compound_runs.csv", index=False)
    sc_df.to_csv(outdir / "benchmark_scenarios_fho_tfho.csv", index=False)

    agg_comp = bench_df.groupby("method").mean(numeric_only=True)
    scagg = sc_df.groupby(["scenario", "method"]).mean(numeric_only=True)
    key_metrics = {
        "compound": {
            "methods": agg_comp[["objective", "fragility", "coverage", "reuse", "cost_norm", "vulnerability", "environment", "penalty"]].round(6).to_dict(orient="index"),
            "tfho_vs_fho": {
                "objective_increase_pct": float(100 * (agg_comp.loc["TFHO", "objective"] / agg_comp.loc["FHO", "objective"] - 1)),
                "fragility_reduction_pct": float(100 * (1 - agg_comp.loc["TFHO", "fragility"] / agg_comp.loc["FHO", "fragility"])),
            },
            "tfho_vs_de": {
                "objective_difference_pct": float(100 * (agg_comp.loc["TFHO", "objective"] / agg_comp.loc["DE", "objective"] - 1)),
                "fragility_reduction_pct": float(100 * (1 - agg_comp.loc["TFHO", "fragility"] / agg_comp.loc["DE", "fragility"])),
            },
            "tfho_vs_ga": {
                "objective_difference_pct": float(100 * (agg_comp.loc["TFHO", "objective"] / agg_comp.loc["GA", "objective"] - 1)),
                "fragility_reduction_pct": float(100 * (1 - agg_comp.loc["TFHO", "fragility"] / agg_comp.loc["GA", "fragility"])),
            },
        },
        "scenario_sensitivity": {},
    }
    for sc in ["baseline", "pollution", "compound"]:
        fho = scagg.loc[(sc, "FHO")]
        tfho = scagg.loc[(sc, "TFHO")]
        key_metrics["scenario_sensitivity"][sc] = {
            "objective_increase_pct": float(100 * (tfho.objective / fho.objective - 1)),
            "fragility_reduction_pct": float(100 * (1 - tfho.fragility / fho.fragility)),
            "coverage_difference": float(tfho.coverage - fho.coverage),
            "reuse_difference": float(tfho.reuse - fho.reuse),
        }
    with open(outdir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(key_metrics, f, indent=2)

    syn = generate_synthetic_system(seed)
    months = syn["months"]
    b_demand = aggregate_bimonthly(syn["demands_m"], "sum")
    b_avail = aggregate_bimonthly(syn["avail_m"], "sum")
    periods = np.arange(1, 7)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    ax = axes[0, 0]
    ax.plot(months, syn["rain"], marker="o")
    ax.set_title("Monthly rainfall signal")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm)")
    ax = axes[0, 1]
    ax.plot(months, syn["temp"], marker="o")
    ax.set_title("Monthly air temperature")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax = axes[1, 0]
    for j, name in enumerate(SECTOR_NAMES):
        ax.plot(periods, b_demand[:, j], marker="o", label=name)
    ax.set_title("Bi-monthly sectoral demand")
    ax.set_xlabel("Bi-monthly period")
    ax.set_ylabel("Demand (10^3 m$^3$)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax = axes[1, 1]
    for s, name in enumerate(SOURCE_NAMES):
        ax.plot(periods, b_avail[:, s], marker="o", label=name)
    ax.set_title("Bi-monthly source availability")
    ax.set_xlabel("Bi-monthly period")
    ax.set_ylabel("Availability (10^3 m$^3$)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figure_1_synthetic_drivers.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    def plot_triangular(ax, low, modal, high, title, xlabel):
        xs = np.linspace(low * 0.9, high * 1.1, 400)
        mu = np.zeros_like(xs)
        left = (xs >= low) & (xs <= modal)
        right = (xs >= modal) & (xs <= high)
        mu[left] = (xs[left] - low) / (modal - low)
        mu[right] = (high - xs[right]) / (high - modal)
        ax.plot(xs, mu)
        centroid = (low + modal + high) / 3
        ax.axvline(centroid, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Membership")
        ax.set_ylim(0, 1.05)

    base = build_case_data("baseline", use_fuzzy=False, seed=seed)
    a_m = base["A"][0, 2]
    d_m = base["D"][0, 2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    plot_triangular(axes[0], A_LOW[0, 0] * a_m, a_m, A_UP[0, 0] * a_m, "Reservoir availability fuzzy number (P3)", "Availability (10^3 m$^3$)")
    plot_triangular(axes[1], D_LOW[0, 0] * d_m, d_m, D_UP[0, 0] * d_m, "Residential demand fuzzy number (P3)", "Demand (10^3 m$^3$)")
    fig.tight_layout()
    fig.savefig(outdir / "figure_2_fuzzy_memberships.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    def history_to_series(history, max_eval=600):
        arr = np.zeros(max_eval)
        cur = history[0][1]
        idx = 0
        for e, val in history:
            while idx < min(e, max_eval):
                arr[idx] = cur
                idx += 1
            cur = val
        while idx < max_eval:
            arr[idx] = cur
            idx += 1
        return arr

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in ["DE", "GA", "FHO"]:
        series = np.vstack([history_to_series(h, budget) for h in histories[method]])
        med = np.median(series, axis=0)
        ax.plot(np.arange(1, budget + 1), med, label=method)
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Best-so-far objective")
    ax.set_title("Compound scenario: median convergence profile")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figure_3_convergence_compound.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    agg = bench_df.groupby("method").agg({"objective": "mean", "fragility": "mean", "runtime": "mean"}).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sizes = 1200 * agg["runtime"] / agg["runtime"].max() + 80
    for i, r in agg.iterrows():
        ax.scatter(r["fragility"], r["objective"], s=sizes.iloc[i], alpha=0.7)
        ax.annotate(r["method"], (r["fragility"], r["objective"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Topological fragility index")
    ax.set_ylabel("Mean objective")
    ax.set_title("Compound scenario trade-off: scalar performance vs topology")
    fig.tight_layout()
    fig.savefig(outdir / "figure_4_tradeoff_compound.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    topo_f = scenario_topology18(ev2.best["x"], data_comp)
    topo_t = scenario_topology18(tf2["x"], data_comp)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, topo, title in [(axes[0, 0], topo_f, "FHO persistence diagram (H1, compound seed 2)"), (axes[0, 1], topo_t, "TFHO persistence diagram (H1, compound seed 2)")]:
        if topo["h1"]:
            births = [b for b, d in topo["h1"]]
            deaths = [d for b, d in topo["h1"]]
            ax.scatter(births, deaths)
        maxv = max([d for _, d in topo["h1"]], default=1.0) * 1.05
        ax.plot([0, maxv], [0, maxv], linestyle="--")
        ax.set_xlim(0, maxv)
        ax.set_ylim(0, maxv)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title)
    for ax, topo, title in [(axes[1, 0], topo_f, "FHO persistence landscape"), (axes[1, 1], topo_t, "TFHO persistence landscape")]:
        xg, lams = persistence_landscape(topo["h1"], k_max=3)
        for k in range(lams.shape[0]):
            ax.plot(xg, lams[k], label=f"λ{k + 1}")
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Landscape amplitude")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "figure_5_persistence_compound_seed2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    def heatmap_alloc(ax, mat, row_labels, col_labels, title):
        im = ax.imshow(mat, aspect="auto")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, color="black")
        return im

    mat_f = ev2.best["x"].sum(axis=2)
    mat_t = tf2["x"].sum(axis=2)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    im = heatmap_alloc(axes[0], mat_f, SOURCE_NAMES, SECTOR_NAMES, "FHO annual allocation totals")
    heatmap_alloc(axes[1], mat_t, SOURCE_NAMES, SECTOR_NAMES, "TFHO annual allocation totals")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("10^3 m$^3$")
    fig.savefig(outdir / "figure_6_allocation_heatmaps_seed2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    sc_mean = sc_df.groupby(["scenario", "method"]).mean(numeric_only=True).reset_index()
    scenarios = ["baseline", "pollution", "compound"]
    x = np.arange(len(scenarios))
    width = 0.35
    frag_f = [sc_mean[(sc_mean.scenario == sc) & (sc_mean.method == "FHO")]["fragility"].iloc[0] for sc in scenarios]
    frag_t = [sc_mean[(sc_mean.scenario == sc) & (sc_mean.method == "TFHO")]["fragility"].iloc[0] for sc in scenarios]
    obj_f = [sc_mean[(sc_mean.scenario == sc) & (sc_mean.method == "FHO")]["objective"].iloc[0] for sc in scenarios]
    obj_t = [sc_mean[(sc_mean.scenario == sc) & (sc_mean.method == "TFHO")]["objective"].iloc[0] for sc in scenarios]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.bar(x - width / 2, frag_f, width, label="FHO")
    ax.bar(x + width / 2, frag_t, width, label="TFHO")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel("Mean topological fragility")
    ax.set_title("Scenario sensitivity of structural fragility")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.bar(x - width / 2, obj_f, width, label="FHO")
    ax.bar(x + width / 2, obj_t, width, label="TFHO")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel("Mean objective")
    ax.set_title("Objective stability under scenario shifts")
    fig.tight_layout()
    fig.savefig(outdir / "figure_7_scenario_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig_paths = [outdir / "figure_1_synthetic_drivers.png", outdir / "figure_2_fuzzy_memberships.png", outdir / "figure_3_convergence_compound.png",
                 outdir / "figure_4_tradeoff_compound.png", outdir / "figure_5_persistence_compound_seed2.png", outdir / "figure_6_allocation_heatmaps_seed2.png",
                 outdir / "figure_7_scenario_bars.png"]
    thumbs = []
    for p in fig_paths:
        img = Image.open(p).convert("RGB")
        img.thumbnail((500, 350))
        canvas = Image.new("RGB", (520, 390), "white")
        canvas.paste(img, ((520 - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 360), p.name, fill="black")
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    montage = Image.new("RGB", (cols * 520, rows * 390), (245, 245, 245))
    for idx, img in enumerate(thumbs):
        montage.paste(img, ((idx % cols) * 520, (idx // cols) * 390))
    montage.save(outdir / "montage.png")

    return {"benchmark_compound": bench_df, "scenario_benchmark": sc_df, "summary": key_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="tfho_case_outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic-data seed")
    parser.add_argument("--budget", type=int, default=600, help="Evaluation budget per stochastic algorithm")
    parser.add_argument("--pop_size", type=int, default=24, help="Population size")
    args = parser.parse_args()
    results = generate_outputs(Path(args.outdir), seed=args.seed, budget=args.budget, pop_size=args.pop_size)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
