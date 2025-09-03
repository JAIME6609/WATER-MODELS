# -*- coding: utf-8 -*-
"""
Monolithic file: withtains the entire base model + feasible PLUS.
from __future__ import annotations
"""
# -*- coding: utf-8 -*-
"""
WATER-MODEL-FL-MM-01_maximal.py

Maximum, self-withtained version, with 26 Metaheuristics, explicit capacity sub-limits,
extended withstraints (Budget, Infrastructure, Micropollutants, Monitoring,
Maintenance, educación), duplicated outputs (PNG, PDF, SVG) y withvergence logs.

Includes:
- Reproducibility (global seeds + RNG por algoritmo)
- Triangular fuzzy logic and defuzzification
- Normalized penalties and violation counts per type
- Global and pair-specific bounds (i,j) + repair
- evaluation cache
- Comparative table, exports CSV/NPY
- Charts: bars (fitness, cost, penalty, time), multi-line optimal vectors,
  stacked-bar of violations, membership demo, convergence curves (by algorithm and combined)
- Save figures in PNG, PDF y SVG

Metaheuristics (26): GA, PSO, DE, SA, HS, ABC, VNS, TS, ES, EP, FA, CS, WOA, BA, TLBO, MA,
                      GSA, ICA, BBO, ALO, QPSO, DA, ACOR, GWO, SCA, HHO.

Author: Comprehensive adaptation with improvements
Date: 2025-08-08
"""

import math
import time
import random
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Callable
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0) Reproducibility y utilities
# ============================================================

def fix_all_seeds(seed: int = 20250808):
    random.seed(seed)
    np.random.seed(seed)

def make_rngs(base_seed: int, names: List[str]) -> Dict[str, np.random.Generator]:
    rngs = {}
    for k, name in enumerate(names, start=1):
        seed = (abs(hash(name)) + 10_000 * k + base_seed) % (2**32 - 1)
        rngs[name] = np.random.default_rng(seed)
    return rngs

def ensure_dirs(base: pathlib.Path) -> Dict[str, pathlib.Path]:
    base.mkdir(parents=True, exist_ok=True)
    d_figs = base / "figs"; d_figs.mkdir(exist_ok=True)
    d_logs = base / "logs"; d_logs.mkdir(exist_ok=True)
    return {"base": base, "figs": d_figs, "logs": d_logs}

def savefig_all(figpath_noext: pathlib.Path):
    """save the current figure in PNG, PDF y SVG."""
    plt.savefig(str(figpath_noext) + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(str(figpath_noext) + ".pdf", bbox_inches="tight")
    plt.savefig(str(figpath_noext) + ".svg", bbox_inches="tight")
    plt.close()

# ============================================================
# 1) Fuzzy logic (triangular) y defuzzification
# ============================================================

def triangular_centroid(a: float, m: float, b: float) -> float:
    return (a + m + b) / 3.0

@dataclass
class FuzzyTri:
    a: float
    m: float
    b: float
    def defuzz(self) -> float:
        return triangular_centroid(self.a, self.m, self.b)

# ============================================================
# 2) Model configuration
# ============================================================

@dataclass
class ModelConfig:
    # Sizes
    n_sources: int = 3
    n_sectors: int = 3
    n_periods: int = 2

    # Reproducibility
    base_seed: int = 20250808

    # Outputs
    results_dir: pathlib.Path = pathlib.Path("results")

    # Penalties and weights
    pen_factor: float = 1e6
    env_frac: float = 0.90                 # environmental limit (availability fraction)
    demand_coverage_min: float = 1.00      # minimum coverage by sector
    drought_resilience_frac: float = 0.90  # total ≥ fraction of total demand (per period)
    strategic_reserve_frac: float = 0.10   # minimum by sector (fraction of demand)

    # Relative weights (all restrictions)
    w_env: float = 1.0
    w_demand: float = 1.0
    w_capacity_total: float = 1.0
    w_capacity_pre: float = 1.0
    w_capacity_primary: float = 1.0
    w_capacity_secondary: float = 1.0
    w_capacity_tertiary: float = 1.0
    w_pair_bound: float = 1.0
    w_nonneg: float = 1.0
    w_drought: float = 1.0
    w_strategic: float = 1.0
    w_budget: float = 1.0
    w_infra: float = 1.0
    w_micro: float = 1.0
    w_monitor: float = 1.0
    w_maint: float = 1.0
    w_edu: float = 1.0

    # Bounds
    global_upper_bound: float = 60_000.0
    pair_upper_bounds: np.ndarray = field(default_factory=lambda: np.full((3,3), 50_000.0))

    # Metaheuristics hyperparameters
    pop_size: int = 60
    n_generations: int = 120

# ============================================================
# 3) Fuzzy data, costs and extended parameters
# ============================================================

def build_fuzzy_data(cfg: ModelConfig):
    # Availability per source (triangular, base period)
    avail_base = [
        FuzzyTri(40_000, 50_000, 60_000),  # superficial
        FuzzyTri(30_000, 45_000, 55_000),  # underground
        FuzzyTri(15_000, 25_000, 35_000),  # reuse
    ]
    # Demand per sector (urban, agricultural, industrial)
    dem_base = [
        FuzzyTri(35_000, 40_000, 50_000),
        FuzzyTri(20_000, 25_000, 30_000),
        FuzzyTri(10_000, 15_000, 22_000),
    ]
    # Costs per source [$/m^3]
    cost_tri = [
        FuzzyTri(0.18, 0.20, 0.23),
        FuzzyTri(0.25, 0.28, 0.33),
        FuzzyTri(0.30, 0.35, 0.42),
    ]
    # Total treatment capacity (diffuse)
    treat_tri = FuzzyTri(55_000, 60_000, 65_000)

    # Scaling by period
    scale_av = np.array([1.00, 1.02])[:cfg.n_periods]
    scale_dm = np.array([1.00, 1.03])[:cfg.n_periods]

    # defuzzification and scaling
    avail = np.zeros((cfg.n_periods, cfg.n_sources))
    for i in range(cfg.n_sources):
        a = avail_base[i].defuzz()
        for t in range(cfg.n_periods):
            avail[t, i] = a * scale_av[t]

    demand = np.zeros((cfg.n_periods, cfg.n_sectors))
    for j in range(cfg.n_sectors):
        d = dem_base[j].defuzz()
        for t in range(cfg.n_periods):
            demand[t, j] = d * scale_dm[t]

    cost = np.array([ct.defuzz() for ct in cost_tri])
    treat_cap = np.array([treat_tri.defuzz() for _ in range(cfg.n_periods)])

    # explicit capacity sub-limits (pedagogical example, adjustable)
    pre_cap = 0.95 * treat_cap
    primary_cap = 0.90 * treat_cap
    secondary_cap = 0.85 * treat_cap
    tertiary_cap = 0.80 * treat_cap

    # minimum (absolute) strategic reserves
    strategic_min = cfg.strategic_reserve_frac * demand

    # -------------------------------
    # Extended parameters
    # -------------------------------
    # Budget (per period) [USD]
    budget_max = np.array([35_000.0, 36_000.0])[:cfg.n_periods]

    # Transport/infrastructure costs per pair (i,j) [USD/m^3]
    trans_cost = np.array([
        [0.04, 0.05, 0.06],
        [0.05, 0.04, 0.06],
        [0.06, 0.05, 0.04],
    ])

    # Infrastructure coefficients (i,j) (dimensionless)
    trans_coef = np.array([
        [0.8, 1.1, 1.0],
        [0.9, 0.9, 1.2],
        [1.2, 1.0, 0.8],
    ])
    # Infrastructure capacity per period (global limit)
    infra_cap = np.array([40_000.0, 41_000.0])[:cfg.n_periods]

    # Quality (micropollutants): with concentration by source [mg/L] and limits per sector
    micro_conc_source = np.array([0.9, 1.2, 0.5])
    micro_max_sector = np.array([1.0, 1.5, 0.8])  # urban, agricultural, industrial

    # Monitoring, maintenance, education (weights and minimums)
    qmon_weight = np.array([
        [0.6, 0.7, 0.8],
        [0.7, 0.6, 0.8],
        [0.8, 0.7, 0.6],
    ])
    maint_weight = np.array([
        [0.5, 0.6, 0.7],
        [0.6, 0.5, 0.7],
        [0.7, 0.6, 0.5],
    ])
    edu_weight_sector = np.array([0.8, 0.6, 0.7])  # weights flow by sector for education
    qmon_min = np.array([8_000.0, 8_200.0])[:cfg.n_periods]
    maint_min = np.array([6_000.0, 6_100.0])[:cfg.n_periods]
    edu_min = np.array([5_000.0, 5_200.0])[:cfg.n_periods]

    return {
        "avail": avail,                     # (T,I)
        "demand": demand,                   # (T,J)
        "cost": cost,                       # (I,)
        "treat_cap": treat_cap,             # (T,)
        "pre_cap": pre_cap,                 # (T,)
        "primary_cap": primary_cap,         # (T,)
        "sewithdary_cap": secondary_cap,     # (T,)
        "tertiary_cap": tertiary_cap,       # (T,)
        "strategic_min": strategic_min,     # (T,J)
        "budget_max": budget_max,           # (T,)
        "trans_cost": trans_cost,           # (I,J)
        "trans_coef": trans_coef,           # (I,J)
        "infra_cap": infra_cap,             # (T,)
        "micro_withc_source": micro_conc_source,  # (I,)
        "micro_max_sector": micro_max_sector,    # (J,)
        "qmon_weight": qmon_weight,         # (I,J)
        "maint_weight": maint_weight,       # (I,J)
        "edu_weight_sector": edu_weight_sector,  # (J,)
        "qmon_min": qmon_min,               # (T,)
        "maint_min": maint_min,             # (T,)
        "edu_min": edu_min,                 # (T,)
    }

# ============================================================
# 4) Indexer and bound repair
# ============================================================

@dataclass
class Indexer:
    I: int
    J: int
    T: int
    def nvars(self) -> int:
        return self.I * self.J * self.T
    def to_tensor(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.I, self.J, self.T, order="F")
    def to_vector(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(self.I * self.J * self.T, order="F")

def clamp_vector_to_bounds(x: np.ndarray, cfg: ModelConfig, idx: Indexer) -> np.ndarray:
    X = idx.to_tensor(x.copy())
    for i in range(idx.I):
        for j in range(idx.J):
            ub = min(cfg.pair_upper_bounds[i, j], cfg.global_upper_bound)
            for t in range(idx.T):
                if X[i, j, t] < 0.0:
                    X[i, j, t] = 0.0
                elif X[i, j, t] > ub:
                    X[i, j, t] = ub
    return idx.to_vector(X)

# ============================================================
# 5) Evaluation with cache, normalized penalties and violations
# ============================================================

@dataclass
class EvalResult:
    fitness: float
    cost: float
    penalty: float
    n_violations: int
    violations_detail: Dict[str, int]

class ModelEvaluator:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.data = build_fuzzy_data(cfg)
        self.cache: Dict[Tuple[float, ...], EvalResult] = {}

        # Normalizers (avoid numerical dominance)
        avail = self.data["avail"]; demand = self.data["demand"]
        self.norm_env = np.maximum(cfg.env_frac * avail, 1.0)       # (T,I)
        self.norm_dem = np.maximum(cfg.demand_coverage_min * demand, 1.0)  # (T,J)
        self.norm_cap_total = np.maximum(self.data["treat_cap"], 1.0)
        self.norm_cap_pre = np.maximum(self.data["pre_cap"], 1.0)
        self.norm_cap_primary = np.maximum(self.data["primary_cap"], 1.0)
        self.norm_cap_secondary = np.maximum(self.data["sewithdary_cap"], 1.0)
        self.norm_cap_tertiary = np.maximum(self.data["tertiary_cap"], 1.0)
        self.norm_pair = np.maximum(cfg.pair_upper_bounds, 1.0)
        self.norm_drought = np.maximum(demand.sum(axis=1), 1.0)
        self.norm_strat = np.maximum(self.data["strategic_min"], 1.0)
        self.norm_budget = np.maximum(self.data["budget_max"], 1.0)
        self.norm_infra = np.maximum(self.data["infra_cap"], 1.0)
        self.norm_qmon = np.maximum(self.data["qmon_min"], 1.0)
        self.norm_maint = np.maximum(self.data["maint_min"], 1.0)
        self.norm_edu = np.maximum(self.data["edu_min"], 1.0)
        # For micro: normalize by sector limit for each j
        self.norm_micro = np.maximum(self.data["micro_max_sector"], 1e-6)

    def _key(self, x: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(x, 6).tolist())

    def evaluate(self, x: np.ndarray, idx: Indexer) -> EvalResult:
        key = self._key(x)
        if key in self.cache:
            return self.cache[key]

        cfg = self.cfg
        D = self.data
        X = idx.to_tensor(x)  # (I,J,T)

        # Pure cost (operation + transport): sum_{i,j,t} (c_i + trans_cost_ij) * x_ijt
        base_cost = 0.0
        for i in range(idx.I):
            for j in range(idx.J):
                unit = D["cost"][i] + D["trans_cost"][i, j]
                base_cost += unit * np.sum(X[i, j, :])

        penalty = 0.0
        vio = defaultdict(int)

        # (1) Environmental per source and period: sum_j x_{i,j,t} <= env_frac * avail[t,i]
        for t in range(idx.T):
            for i in range(idx.I):
                used = np.sum(X[i, :, t])
                limit = cfg.env_frac * D["avail"][t, i]
                if used > limit:
                    v = (used - limit) / self.norm_env[t, i]
                    penalty += cfg.pen_factor * cfg.w_env * (v ** 2)
                    vio["ambiental"] += 1

        # (2) Minimum demand per sector and period: sum_i x_{i,j,t} >= demand_min
        for t in range(idx.T):
            for j in range(idx.J):
                supplied = np.sum(X[:, j, t])
                demand_min = cfg.demand_coverage_min * D["demand"][t, j]
                if supplied < demand_min:
                    v = (demand_min - supplied) / self.norm_dem[t, j]
                    penalty += cfg.pen_factor * cfg.w_demand * (v ** 2)
                    vio["demand_min"] += 1

        # (3) Explicit capacities (total and sub-stages) per period
        for t in range(idx.T):
            total_t = np.sum(X[:, :, t])
            # Total
            cap = D["treat_cap"][t]
            if total_t > cap:
                v = (total_t - cap) / self.norm_cap_total[t]
                penalty += cfg.pen_factor * cfg.w_capacity_total * (v ** 2)
                vio["cap_total"] += 1
            # Pretreatment
            if total_t > D["pre_cap"][t]:
                v = (total_t - D["pre_cap"][t]) / self.norm_cap_pre[t]
                penalty += cfg.pen_factor * cfg.w_capacity_pre * (v ** 2)
                vio["cap_pre"] += 1
            # Primary
            if total_t > D["primary_cap"][t]:
                v = (total_t - D["primary_cap"][t]) / self.norm_cap_primary[t]
                penalty += cfg.pen_factor * cfg.w_capacity_primary * (v ** 2)
                vio["cap_prim"] += 1
            # Sewithdary
            if total_t > D["sewithdary_cap"][t]:
                v = (total_t - D["sewithdary_cap"][t]) / self.norm_cap_secondary[t]
                penalty += cfg.pen_factor * cfg.w_capacity_secondary * (v ** 2)
                vio["cap_sec"] += 1
            # Tertiary
            if total_t > D["tertiary_cap"][t]:
                v = (total_t - D["tertiary_cap"][t]) / self.norm_cap_tertiary[t]
                penalty += cfg.pen_factor * cfg.w_capacity_tertiary * (v ** 2)
                vio["cap_terc"] += 1

        # (4) Bounds por par (i,j) y non-negativity
        for i in range(idx.I):
            for j in range(idx.J):
                ub = min(cfg.pair_upper_bounds[i, j], cfg.global_upper_bound)
                for t in range(idx.T):
                    if X[i, j, t] > ub:
                        v = (X[i, j, t] - ub) / self.norm_pair[i, j]
                        penalty += cfg.pen_factor * cfg.w_pair_bound * (v ** 2)
                        vio["cota_par"] += 1
                    if X[i, j, t] < 0.0:
                        v = (-X[i, j, t]) / self.norm_pair[i, j]
                        penalty += cfg.pen_factor * cfg.w_nonneg * (v ** 2)
                        vio["no_negatividad"] += 1

        # (5) Drought resilience: total_t >= drought_frac * sum_j demand[t,j]
        for t in range(idx.T):
            total_t = np.sum(X[:, :, t])
            min_total = cfg.drought_resilience_frac * np.sum(D["demand"][t, :])
            if total_t < min_total:
                v = (min_total - total_t) / self.norm_drought[t]
                penalty += cfg.pen_factor * cfg.w_drought * (v ** 2)
                vio["sequía"] += 1

        # (6) strategic reserves (≥) by sector and period
        for t in range(idx.T):
            for j in range(idx.J):
                supplied = np.sum(X[:, j, t])
                min_res = D["strategic_min"][t, j]
                if supplied < min_res:
                    v = (min_res - supplied) / self.norm_strat[t, j]
                    penalty += cfg.pen_factor * cfg.w_strategic * (v ** 2)
                    vio["reserva_estratégica"] += 1

        # (7) Budget: sum_{i,j} (c_i + trans_cost_ij)*x_ijt <= budget_max[t]
        for t in range(idx.T):
            cost_t = 0.0
            for i in range(idx.I):
                for j in range(idx.J):
                    unit = D["cost"][i] + D["trans_cost"][i, j]
                    cost_t += unit * X[i, j, t]
            if cost_t > D["budget_max"][t]:
                v = (cost_t - D["budget_max"][t]) / self.norm_budget[t]
                penalty += cfg.pen_factor * cfg.w_budget * (v ** 2)
                vio["Budget"] += 1

        # (8) Infrastructure: sum_{i,j} trans_coef_ij * x_ijt <= infra_cap[t]
        for t in range(idx.T):
            infra_load = 0.0
            for i in range(idx.I):
                for j in range(idx.J):
                    infra_load += D["trans_coef"][i, j] * X[i, j, t]
            if infra_load > D["infra_cap"][t]:
                v = (infra_load - D["infra_cap"][t]) / self.norm_infra[t]
                penalty += cfg.pen_factor * cfg.w_infra * (v ** 2)
                vio["Infrastructure"] += 1

        # (9) Micropollutants: sector-weighted average <= micro_max_sector[j]
        #   c_j(t) = (sum_i withc_i * x_ijt) / (sum_i x_ijt) <= micro_max[j]
        for t in range(idx.T):
            for j in range(idx.J):
                denom = np.sum(X[:, j, t])
                if denom > 0:
                    numer = np.sum(D["micro_withc_source"] * X[:, j, t])
                    c_j = numer / denom
                    if c_j > D["micro_max_sector"][j]:
                        v = (c_j - D["micro_max_sector"][j]) / self.norm_micro[j]
                        penalty += cfg.pen_factor * cfg.w_micro * (v ** 2)
                        vio["Micropollutants"] += 1
                else:
                    # If there is no supply, there is no penalty here (it is already penalized by demand/reserve)
                    pass

        # (10) Monitoring: sum_{i,j} qmon_weight_ij * x_ijt >= qmon_min[t]
        for t in range(idx.T):
            mon = 0.0
            for i in range(idx.I):
                for j in range(idx.J):
                    mon += D["qmon_weight"][i, j] * X[i, j, t]
            if mon < D["qmon_min"][t]:
                v = (D["qmon_min"][t] - mon) / self.norm_qmon[t]
                penalty += cfg.pen_factor * cfg.w_monitor * (v ** 2)
                vio["Monitoring"] += 1

        # (11) Maintenance: sum_{i,j} maint_weight_ij * x_ijt >= maint_min[t]
        for t in range(idx.T):
            mnt = 0.0
            for i in range(idx.I):
                for j in range(idx.J):
                    mnt += D["maint_weight"][i, j] * X[i, j, t]
            if mnt < D["maint_min"][t]:
                v = (D["maint_min"][t] - mnt) / self.norm_maint[t]
                penalty += cfg.pen_factor * cfg.w_maint * (v ** 2)
                vio["Maintenance"] += 1

        # (12) Education/awareness: sum_j (edu_weight_j * sum_i x_ijt) >= edu_min[t]
        for t in range(idx.T):
            edu = 0.0
            for j in range(idx.J):
                edu += D["edu_weight_sector"][j] * np.sum(X[:, j, t])
            if edu < D["edu_min"][t]:
                v = (D["edu_min"][t] - edu) / self.norm_edu[t]
                penalty += cfg.pen_factor * cfg.w_edu * (v ** 2)
                vio["educación"] += 1

        fitness = base_cost + penalty
        out = EvalResult(
            fitness=fitness,
            cost=base_cost,
            penalty=penalty,
            n_violations=sum(vio.values()),
            violations_detail=dict(vio)
        )
        self.cache[key] = out
        return out

    def fitness_only(self, x: np.ndarray, idx: Indexer) -> float:
        return self.evaluate(x, idx).fitness

# ============================================================
# 6) Generation, repair and initialization
# ============================================================
# ================================================================
# === FEASIBLE-FIRST wrapper with adaptive penalty    ===
# ================================================================
class FeasibleFirstAdaptiveEvaluator(ModelEvaluator):
    """Prioritize feasible solutions; if not, apply large-M and adaptive α factor."""
    def __init__(self, cfg: ModelConfig, alpha_min: float=0.5, alpha_max: float=5.0,
                 window: int=20, big_M: float=1e6):
        super().__init__(cfg)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha = float(alpha_min)
        self.window = int(max(5, window))
        self.big_M = float(big_M)
        self._recent_feasible = []

    def _update_alpha(self, is_feasible: bool):
        self._recent_feasible.append(1 if is_feasible else 0)
        if len(self._recent_feasible) > self.window:
            self._recent_feasible.pop(0)
        rate = sum(self._recent_feasible) / max(1, len(self._recent_feasible))
        if rate < 0.5:
            self.alpha = min(self.alpha_max, self.alpha * 1.1)
        elif rate > 0.7:
            self.alpha = max(self.alpha_min, self.alpha * 0.9)

    def fitness_only(self, x_vec, idx: Indexer):
        ev = self.evaluate(x_vec, idx)
        self._update_alpha(ev.penalty == 0.0)
        if ev.penalty <= 0.0:
            return ev.cost
        return ev.cost + self.big_M + self.alpha * (ev.penalty)


def random_solution(cfg: ModelConfig, idx: Indexer, rng: np.random.Generator) -> np.ndarray:
    X = np.zeros((idx.I, idx.J, idx.T), dtype=float)
    for i in range(idx.I):
        for j in range(idx.J):
            ub = min(cfg.pair_upper_bounds[i, j], cfg.global_upper_bound)
            for t in range(idx.T):
                X[i, j, t] = rng.uniform(0.0, ub)
    return idx.to_vector(X)

def repair(x: np.ndarray, cfg: ModelConfig, idx: Indexer) -> np.ndarray:
    return clamp_vector_to_bounds(x, cfg, idx)

# ============================================================
# 7) Metaheuristics (all return: best_x, best_f, history_list)
# ============================================================

# To save space: some functions share templates

def wrap_return(best: np.ndarray, fbest: float, hist: List[float]):
    return best, float(fbest), list(hist)

# --- GA ---
def GA_optimize(evaluator, cfg, idx, rng):
    pop, gens = cfg.pop_size, cfg.n_generations
    p_mut, p_xov = 0.20, 0.80
    sigma = 0.10 * cfg.global_upper_bound
    P = np.vstack([random_solution(cfg, idx, rng) for _ in range(pop)])
    F = np.array([evaluator.fitness_only(x, idx) for x in P])
    ib = int(np.argmin(F)); best, fbest = P[ib].copy(), float(F[ib])
    hist = [fbest]

    def tournament(k=3):
        cand = rng.integers(0, pop, size=k)
        return P[cand[np.argmin(F[cand])]].copy()

    for _ in range(gens):
        newP = [best.copy()]
        while len(newP) < pop:
            if rng.random() < p_xov:
                p1, p2 = tournament(), tournament()
                child = np.where(rng.random(p1.size)<0.5, p1, p2)
            else:
                child = tournament()
            if rng.random() < p_mut:
                child += rng.normal(0.0, sigma, size=child.size)
            child = repair(child, cfg, idx)
            newP.append(child)
        P = np.vstack(newP[:pop])
        F = np.array([evaluator.fitness_only(x, idx) for x in P])
        ib = int(np.argmin(F))
        if F[ib] < fbest:
            best, fbest = P[ib].copy(), float(F[ib])
        hist.append(fbest)
    return wrap_return(best, fbest, hist)

# --- PSO ---
def PSO_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    w, c1, c2 = 0.72, 1.5, 1.5
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    V = rng.normal(0.0, 0.05*cfg.global_upper_bound, size=X.shape)
    Pbest = X.copy()
    Fp = np.array([evaluator.fitness_only(x, idx) for x in X])
    g = int(np.argmin(Fp)); Gbest, Fg = X[g].copy(), float(Fp[g])
    hist = [Fg]
    for _ in range(gens):
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w*V + c1*r1*(Pbest-X) + c2*r2*(Gbest-X)
        X = np.vstack([repair(x+v, cfg, idx) for x,v in zip(X,V)])
        F = np.array([evaluator.fitness_only(x, idx) for x in X])
        imp = F < Fp
        Pbest[imp], Fp[imp] = X[imp], F[imp]
        g = int(np.argmin(Fp))
        if Fp[g] < Fg: Gbest, Fg = Pbest[g].copy(), float(Fp[g])
        hist.append(Fg)
    return wrap_return(Gbest, Fg, hist)

# --- DE ---
def DE_optimize(evaluator, cfg, idx, rng):
    NP, gens = cfg.pop_size, cfg.n_generations
    Fm, CR = 0.8, 0.9
    P = np.vstack([random_solution(cfg, idx, rng) for _ in range(NP)])
    Fvals = np.array([evaluator.fitness_only(x, idx) for x in P])
    hist = [float(Fvals.min())]
    for _ in range(gens):
        for i in range(NP):
            idxs = list(range(NP)); idxs.remove(i)
            a, b, c = rng.choice(idxs, 3, replace=False)
            v = P[a] + Fm*(P[b]-P[c])
            mask = rng.random(v.size) < CR; mask[rng.integers(0, v.size)] = True
            u = np.where(mask, v, P[i])
            u = repair(u, cfg, idx)
            Fu = evaluator.fitness_only(u, idx)
            if Fu < Fvals[i]: P[i], Fvals[i] = u, Fu
        hist.append(float(Fvals.min()))
    ib = int(np.argmin(Fvals))
    return wrap_return(P[ib].copy(), float(Fvals[ib]), hist)

# --- SA ---
def SA_optimize(evaluator, cfg, idx, rng):
    x = random_solution(cfg, idx, rng); fx = evaluator.fitness_only(x, idx)
    best, fbest = x.copy(), float(fx)
    T, alpha, sigma = 1_000.0, 0.99, 0.10*cfg.global_upper_bound
    iters_T = max(30, idx.nvars()//2)
    hist = [fbest]
    while T > 1e-3:
        for _ in range(iters_T):
            cand = repair(x + rng.normal(0.0, sigma, x.size), cfg, idx)
            fc = evaluator.fitness_only(cand, idx)
            if fc < fx or rng.random() < math.exp(-(fc - fx)/T):
                x, fx = cand, fc
                if fx < fbest: best, fbest = x.copy(), float(fx)
        T *= alpha; sigma *= 0.98
        hist.append(fbest)
    return wrap_return(best, fbest, hist)

# --- HS ---
def HS_optimize(evaluator, cfg, idx, rng):
    HMS, NI, HMCR, PAR = cfg.pop_size, cfg.n_generations, 0.9, 0.3
    bw = 0.05*cfg.global_upper_bound
    HM = np.vstack([random_solution(cfg, idx, rng) for _ in range(HMS)])
    F = np.array([evaluator.fitness_only(x, idx) for x in HM])
    hist = [float(F.min())]
    for _ in range(NI):
        x_new = np.zeros(idx.nvars())
        for d in range(idx.nvars()):
            if rng.random() < HMCR:
                k = rng.integers(0, HMS); x_new[d] = HM[k, d]
                if rng.random() < PAR: x_new[d] += rng.uniform(-bw, bw)
            else:
                x_new[d] = rng.uniform(0.0, cfg.global_upper_bound)
        x_new = repair(x_new, cfg, idx)
        f_new = evaluator.fitness_only(x_new, idx)
        worst = int(np.argmax(F))
        if f_new < F[worst]: HM[worst], F[worst] = x_new, f_new
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(HM[ib].copy(), float(F[ib]), hist)

# --- ABC ---
def ABC_optimize(evaluator, cfg, idx, rng):
    SN, gens = cfg.pop_size//2, cfg.n_generations
    def init_food():
        x = random_solution(cfg, idx, rng); return x, evaluator.fitness_only(x, idx)
    foods = [init_food() for _ in range(SN)]
    X = np.vstack([f[0] for f in foods]); F = np.array([f[1] for f in foods])
    trials = np.zeros(SN, dtype=int); limit = max(20, idx.nvars())
    hist = [float(F.min())]
    def neighbor(x, k):
        j = rng.integers(0, x.size); phi = rng.uniform(-1, 1)
        cand = x.copy(); cand[j] += phi*(x[j] - X[k, j])
        return repair(cand, cfg, idx)
    for _ in range(gens):
        for i in range(SN):
            k = rng.integers(0, SN); 
            while k == i: k = rng.integers(0, SN)
            cand = neighbor(X[i], k); fc = evaluator.fitness_only(cand, idx)
            if fc < F[i]: X[i], F[i], trials[i] = cand, fc, 0
            else: trials[i] += 1
        fit_inv = 1.0/(1.0 + F - F.min() + 1e-12); probs = fit_inv/fit_inv.sum()
        for _k in range(SN):
            i = rng.choice(np.arange(SN), p=probs)
            k = rng.integers(0, SN); 
            while k == i: k = rng.integers(0, SN)
            cand = neighbor(X[i], k); fc = evaluator.fitness_only(cand, idx)
            if fc < F[i]: X[i], F[i], trials[i] = cand, fc, 0
            else: trials[i] += 1
        for i in range(SN):
            if trials[i] > limit:
                X[i] = random_solution(cfg, idx, rng)
                F[i] = evaluator.fitness_only(X[i], idx)
                trials[i] = 0
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- VNS ---
def VNS_optimize(evaluator, cfg, idx, rng):
    x = random_solution(cfg, idx, rng); fx = evaluator.fitness_only(x, idx)
    best, fbest = x.copy(), float(fx)
    neighborhoods = [0.01, 0.05, 0.10]
    hist = [fbest]
    for _ in range(cfg.n_generations):
        improved = False
        for frac in neighborhoods:
            sigma = frac*cfg.global_upper_bound
            y = repair(best + rng.normal(0.0, sigma, best.size), cfg, idx)
            fy = evaluator.fitness_only(y, idx)
            if fy < fbest: best, fbest, improved = y, fy, True; break
        if not improved:
            z = random_solution(cfg, idx, rng); fz = evaluator.fitness_only(z, idx)
            if fz < fbest: best, fbest = z, fz
        hist.append(fbest)
    return wrap_return(best.copy(), fbest, hist)

# --- TS ---
def TS_optimize(evaluator, cfg, idx, rng):
    x = random_solution(cfg, idx, rng); fx = evaluator.fitness_only(x, idx)
    best, fbest = x.copy(), float(fx)
    tabu, tabu_size = [], 15
    sigma = 0.05*cfg.global_upper_bound
    hist = [fbest]
    for _ in range(cfg.n_generations):
        neighbors, vals = [], []
        for _k in range(30):
            cand = repair(x + rng.normal(0.0, sigma, x.size), cfg, idx)
            key = tuple(np.round(cand, 6).tolist())
            if key in tabu: continue
            fc = evaluator.fitness_only(cand, idx)
            neighbors.append(cand); vals.append(fc)
        if not neighbors:
            x = random_solution(cfg, idx, rng); fx = evaluator.fitness_only(x, idx)
        else:
            ib = int(np.argmin(vals))
            x, fx = neighbors[ib], float(vals[ib])
            tabu.append(tuple(np.round(x, 6).tolist()))
            if len(tabu) > tabu_size: tabu.pop(0)
        if fx < fbest: best, fbest = x.copy(), fx
        hist.append(fbest)
    return wrap_return(best, fbest, hist)

# --- ES ---
def ES_optimize(evaluator, cfg, idx, rng):
    mu, lam = max(8, cfg.pop_size//4), cfg.pop_size
    gens = cfg.n_generations; sigma0 = 0.1*cfg.global_upper_bound
    parents = np.vstack([random_solution(cfg, idx, rng) for _ in range(mu)])
    sigmas = np.full((mu, parents.shape[1]), sigma0)
    Fp = np.array([evaluator.fitness_only(x, idx) for x in parents])
    hist = [float(Fp.min())]
    for _ in range(gens):
        children, child_sigmas = [], []
        for i in range(mu):
            for _k in range(lam//mu):
                tau, tau0 = 1/np.sqrt(2*parents.shape[1]), 1/np.sqrt(2*np.sqrt(parents.shape[1]))
                s = sigmas[i]*np.exp(tau0*rng.normal() + tau*rng.normal(size=sigmas[i].size))
                y = repair(parents[i] + rng.normal(0.0, s, size=parents[i].size), cfg, idx)
                children.append(y); child_sigmas.append(s)
        C = np.vstack(children); Sc = np.vstack(child_sigmas)
        Fc = np.array([evaluator.fitness_only(x, idx) for x in C])
        sel = np.argsort(Fc)[:mu]
        parents, sigmas, Fp = C[sel], Sc[sel], Fc[sel]
        hist.append(float(Fp.min()))
    ib = int(np.argmin(Fp))
    return wrap_return(parents[ib].copy(), float(Fp[ib]), hist)

# --- EP ---
def EP_optimize(evaluator, cfg, idx, rng):
    N, gens = cfg.pop_size, cfg.n_generations
    P = np.vstack([random_solution(cfg, idx, rng) for _ in range(N)])
    sigmas = np.full_like(P, 0.1*cfg.global_upper_bound)
    F = np.array([evaluator.fitness_only(x, idx) for x in P])
    hist = [float(F.min())]
    for _ in range(gens):
        tau, tau0 = 1/np.sqrt(2*P.shape[1]), 1/np.sqrt(2*np.sqrt(P.shape[1]))
        Off = []
        for i in range(N):
            s = sigmas[i]*np.exp(tau0*rng.normal() + tau*rng.normal(size=sigmas[i].size))
            y = repair(P[i] + rng.normal(0.0, s, size=P[i].size), cfg, idx)
            Off.append(y)
        Off = np.vstack(Off); F_off = np.array([evaluator.fitness_only(x, idx) for x in Off])
        allX = np.vstack([P, Off]); allF = np.hstack([F, F_off])
        sel = np.argsort(allF)[:N]
        P, F = allX[sel], allF[sel]
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(P[ib].copy(), float(F[ib]), hist)

# --- FA ---
def FA_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    beta0, gamma, alpha = 1.0, 1e-4, 0.05*cfg.global_upper_bound
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    for _ in range(gens):
        for i in range(n):
            for j in range(n):
                if F[j] < F[i]:
                    r2 = np.sum((X[i]-X[j])**2)
                    beta = beta0*math.exp(-gamma*r2)
                    step = beta*(X[j]-X[i]) + alpha*rng.normal(0.0, 1.0, size=X[i].size)
                    X[i] = repair(X[i] + step, cfg, idx)
                    F[i] = evaluator.fitness_only(X[i], idx)
        alpha *= 0.99
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- CS ---
def levy_flight(rng, size, beta=1.5, scale=0.01):
    sigma_u = (math.gamma(1+beta)*math.sin(math.pi*beta/2) / (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = rng.normal(0, sigma_u, size=size); v = rng.normal(0, 1, size=size)
    step = u / (np.abs(v)**(1/beta))
    return scale * step

def CS_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    pa = 0.25
    nests = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in nests])
    hist = [float(F.min())]
    for _ in range(gens):
        for i in range(n):
            step = levy_flight(rng, nests[i].size, scale=0.01*cfg.global_upper_bound)
            cand = repair(nests[i] + step, cfg, idx)
            fc = evaluator.fitness_only(cand, idx)
            if fc < F[i]: nests[i], F[i] = cand, fc
        k = int(pa*n); worst_idx = np.argsort(F)[-k:] if k>0 else []
        for i in worst_idx:
            nests[i] = random_solution(cfg, idx, rng)
            F[i] = evaluator.fitness_only(nests[i], idx)
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(nests[ib].copy(), float(F[ib]), hist)

# --- WOA ---
def WOA_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    g = int(np.argmin(F)); Xbest, Fbest = X[g].copy(), float(F[g])
    hist = [Fbest]
    for t in range(gens):
        a = 2 - 2*(t/(gens-1))
        for i in range(n):
            r1, r2 = rng.random(), rng.random()
            A = 2*a*r1 - a; C = 2*r2; p = rng.random()
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C*Xbest - X[i]); X[i] = Xbest - A*D
                else:
                    j = rng.integers(0, n); D = np.abs(C*X[j] - X[i]); X[i] = X[j] - A*D
            else:
                D = np.abs(Xbest - X[i]); b, l = 1, rng.uniform(-1, 1)
                X[i] = D*np.exp(b*l)*np.cos(2*math.pi*l) + Xbest
            X[i] = repair(X[i], cfg, idx); F[i] = evaluator.fitness_only(X[i], idx)
        g = int(np.argmin(F))
        if F[g] < Fbest: Xbest, Fbest = X[g].copy(), float(F[g])
        hist.append(Fbest)
    return wrap_return(Xbest, Fbest, hist)

# --- BA ---
def BA_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    Qmin, Qmax = 0.0, 2.0
    A = np.ones(n) * 0.9; r = np.ones(n) * 0.1
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    V = np.zeros_like(X); F = np.array([evaluator.fitness_only(x, idx) for x in X])
    g = int(np.argmin(F)); Xbest, Fbest = X[g].copy(), float(F[g])
    hist = [Fbest]
    for _ in range(gens):
        Q = Qmin + (Qmax - Qmin)*rng.random(n)
        for i in range(n):
            V[i] += (X[i] - Xbest)*Q[i]
            S = X[i] + V[i]
            if rng.random() > r[i]:
                S = Xbest + rng.normal(0.0, 0.01*cfg.global_upper_bound, size=S.size)
            S = repair(S, cfg, idx)
            FS = evaluator.fitness_only(S, idx)
            if (FS <= F[i]) and (rng.random() < A[i]):
                X[i], F[i] = S, FS
                A[i] *= 0.95; r[i] = r[i]*(1 - 0.01) + 0.01
        g = int(np.argmin(F))
        if F[g] < Fbest: Xbest, Fbest = X[g].copy(), float(F[g])
        hist.append(Fbest)
    return wrap_return(Xbest, Fbest, hist)

# --- TLBO ---
def TLBO_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    for _ in range(gens):
        t_idx = int(np.argmin(F)); T = X[t_idx]; mean = X.mean(axis=0)
        TF = 1 + rng.integers(0, 2)
        for i in range(n):
            V = X[i] + rng.random(X[i].size)*(T*TF - mean)
            V = repair(V, cfg, idx); FV = evaluator.fitness_only(V, idx)
            if FV < F[i]: X[i], F[i] = V, FV
        for i in range(n):
            j = rng.integers(0, n)
            while j == i: j = rng.integers(0, n)
            if F[i] < F[j]:
                V = X[i] + rng.random(X[i].size)*(X[i] - X[j])
            else:
                V = X[i] + rng.random(X[i].size)*(X[j] - X[i])
            V = repair(V, cfg, idx); FV = evaluator.fitness_only(V, idx)
            if FV < F[i]: X[i], F[i] = V, FV
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- MA ---
def MA_optimize(evaluator, cfg, idx, rng):
    best, fbest, hist_ga = GA_optimize(evaluator, cfg, idx, rng)
    sigma = 0.02*cfg.global_upper_bound
    hist = list(hist_ga)
    for _ in range(100):
        cand = repair(best + rng.normal(0.0, sigma, size=best.size), cfg, idx)
        fc = evaluator.fitness_only(cand, idx)
        if fc < fbest:
            best, fbest = cand, fc
            sigma *= 0.99
        else:
            sigma *= 0.995
        hist.append(fbest)
    return wrap_return(best, fbest, hist)

# --- GSA (Gravitational Search Algorithm) ---
def GSA_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    G0 = 100; alpha = 20.0
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    V = np.zeros_like(X)
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    for t in range(1, gens+1):
        best = np.min(F); worst = np.max(F)
        if worst == best: worst = best + 1e-9
        m = (F - worst) / (best - worst)  # valores en [0,1]
        m = np.exp(m) + 1e-12
        M = m / np.sum(m)
        G = G0 * math.exp(-alpha * t / gens)
        acc = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                diff = X[j] - X[i]
                dist = np.linalg.norm(diff) + 1e-9
                force = G * (M[i]*M[j]) * diff / dist
                acc[i] += rng.random() * force
        V = rng.random() * V + acc
        X = np.vstack([repair(x + v, cfg, idx) for x,v in zip(X,V)])
        F = np.array([evaluator.fitness_only(x, idx) for x in X])
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- ICA (Imperialist Competitive Algorithm) ---
def ICA_optimize(evaluator, cfg, idx, rng):
    N = cfg.pop_size; gens = cfg.n_generations
    n_imp = max(3, N//6)  # imperios
    P = np.vstack([random_solution(cfg, idx, rng) for _ in range(N)])
    F = np.array([evaluator.fitness_only(x, idx) for x in P])
    hist = [float(F.min())]
    ord_idx = np.argsort(F)
    imperialists = P[ord_idx[:n_imp]]
    imp_f = F[ord_idx[:n_imp]]
    colonies = P[ord_idx[n_imp:]]
    col_f = F[ord_idx[n_imp:]]
    # simple colony allocation
    assign = np.argmin(np.abs(col_f[:,None] - imp_f[None,:]), axis=1)
    for _ in range(gens):
        # Assimilation: colonies move towards their imperialist
        for k in range(n_imp):
            mask = (assign == k)
            for i, c in enumerate(colonies[mask]):
                step = rng.random(c.size)*(imperialists[k] - c)
                colonies[mask][i] = repair(c + step, cfg, idx)
        # Revolution: small disturbances
        colonies += rng.normal(0.0, 0.01*cfg.global_upper_bound, size=colonies.shape)
        colonies = np.vstack([repair(c, cfg, idx) for c in colonies])
        # Re-evaluate
        imp_f = np.array([evaluator.fitness_only(x, idx) for x in imperialists])
        col_f = np.array([evaluator.fitness_only(x, idx) for x in colonies])
        # Competition: The worst colony can change empire to the best imperialist
        if len(colonies) > 0:
            w = np.argmax(col_f); b = np.argmin(imp_f)
            assign[w] = b
        hist.append(float(min(imp_f.min(), col_f.min() if len(col_f)>0 else np.inf)))
    # Best of all
    allX = np.vstack([imperialists, colonies])
    allF = np.hstack([imp_f, col_f])
    ib = int(np.argmin(allF))
    return wrap_return(allX[ib].copy(), float(allF[ib]), hist)

# --- BBO (Biogeography-Based Optimization) ---
def BBO_optimize(evaluator, cfg, idx, rng):
    N, gens = cfg.pop_size, cfg.n_generations
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(N)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    for _ in range(gens):
        rank = np.argsort(F)
        mu = np.linspace(1.0, 0.0, N)   # immigration
        lam = 1.0 - mu                  # emigration
        X_new = X.copy()
        for i in range(N):
            for d in range(X.shape[1]):
                if rng.random() < mu[i]:
                    k = rng.choice(rank[:max(2, N//3)])
                    X_new[i, d] = X[i, d] + lam[k]*(X[k, d] - X[i, d])*rng.random()
        # mutation
        X_new += rng.normal(0.0, 0.005*cfg.global_upper_bound, size=X_new.shape)
        X_new = np.vstack([repair(x, cfg, idx) for x in X_new])
        F_new = np.array([evaluator.fitness_only(x, idx) for x in X_new])
        sel = F_new < F
        X[sel], F[sel] = X_new[sel], F_new[sel]
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- ALO (Ant Lion Optimizer) ---
def ALO_optimize(evaluator, cfg, idx, rng):
    N, gens = cfg.pop_size, cfg.n_generations
    ants = np.vstack([random_solution(cfg, idx, rng) for _ in range(N)])
    lions = np.vstack([random_solution(cfg, idx, rng) for _ in range(N)])
    F_a = np.array([evaluator.fitness_only(x, idx) for x in ants])
    F_l = np.array([evaluator.fitness_only(x, idx) for x in lions])
    hist = [float(min(F_a.min(), F_l.min()))]
    for _ in range(gens):
        elite = lions[np.argmin(F_l)].copy()
        new_ants = []
        for i in range(N):
            l = rng.integers(0, N)
            LB = np.minimum(ants[i], lions[l]); UB = np.maximum(ants[i], lions[l])
            R = np.cumsum(rng.integers(-1, 2, size=ants[i].size))
            R = (R - R.min()) / (R.max() - R.min() + 1e-9)
            pos = LB + R*(UB - LB)
            pos = repair(pos + 0.1*(elite - pos)*rng.random(), cfg, idx)
            new_ants.append(pos)
        ants = np.vstack(new_ants)
        F_a = np.array([evaluator.fitness_only(x, idx) for x in ants])
        # hunting: replacing worse lions
        allX = np.vstack([lions, ants]); allF = np.hstack([F_l, F_a])
        sel = np.argsort(allF)[:N]
        lions, F_l = allX[sel], allF[sel]
        hist.append(float(F_l.min()))
    ib = int(np.argmin(F_l))
    return wrap_return(lions[ib].copy(), float(F_l[ib]), hist)

# --- QPSO (Quantum-behaved PSO) ---
def QPSO_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    alpha0 = 0.75
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    g = int(np.argmin(F)); Gbest, Fg = X[g].copy(), float(F[g])
    hist = [Fg]
    for t in range(gens):
        alpha = alpha0 * (1 - t/(gens+1))
        mbest = np.mean(X, axis=0)
        for i in range(n):
            u = rng.random(X.shape[1])
            p = (X[i] + Gbest)/2.0
            X[i] = p + alpha * np.sign(u - 0.5) * np.log(1.0/u) * np.abs(mbest - X[i])
            X[i] = repair(X[i], cfg, idx)
            F[i] = evaluator.fitness_only(X[i], idx)
        g = int(np.argmin(F))
        if F[g] < Fg: Gbest, Fg = X[g].copy(), float(F[g])
        hist.append(Fg)
    return wrap_return(Gbest, Fg, hist)

# --- DA (Dragonfly Algorithm) ---
def DA_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    V = rng.normal(0.0, 0.01*cfg.global_upper_bound, size=X.shape)
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    s, a, c = 0.1, 0.1, 0.1  # separation, alignment, cohesion
    for _ in range(gens):
        for i in range(n):
            # random neighborhood
            Nn = rng.choice(n, size=max(3, n//5), replace=False)
            Xn = X[Nn]; Vn = V[Nn]
            S = -np.sum(Xn - X[i], axis=0); A = np.mean(Vn, axis=0); C = (np.mean(Xn, axis=0) - X[i])
            food = X[np.argmin(F)]; enemy = X[np.argmax(F)]
            Fd = food - X[i]; En = X[i] - enemy
            V[i] = rng.random()*V[i] + s*S + a*A + c*C + rng.random()*Fd + rng.random()*En
            X[i] = repair(X[i] + V[i], cfg, idx)
            F[i] = evaluator.fitness_only(X[i], idx)
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- ACOR (Ant Colony Optimization for withtinuous Domains) ---
def ACOR_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    k = max(5, n//3)
    archive = np.vstack([random_solution(cfg, idx, rng) for _ in range(k)])
    F = np.array([evaluator.fitness_only(x, idx) for x in archive])
    hist = [float(F.min())]
    q = 0.5; xi = 0.85
    for _ in range(gens):
        ord_idx = np.argsort(F); archive = archive[ord_idx]; F = F[ord_idx]
        w = np.array([1/(np.sqrt(2*math.pi)*q*k) * math.exp(- (i**2)/(2*(q**2)*k**2)) for i in range(k)])
        w = w / w.sum()
        # deviation per component (sigma) from file
        sigma = np.zeros_like(archive)
        for j in range(archive.shape[1]):
            col = archive[:, j]
            sigma[:, j] = xi * np.abs(col - np.sum(w*col))
        # generate n solutions
        newX = []
        for _r in range(n):
            idx_k = rng.choice(np.arange(k), p=w)
            mu = archive[idx_k]
            s = sigma[idx_k]
            cand = rng.normal(mu, np.maximum(s, 1e-9))
            cand = repair(cand, cfg, idx)
            newX.append(cand)
        newX = np.vstack(newX); F_new = np.array([evaluator.fitness_only(x, idx) for x in newX])
        # update file
        allX = np.vstack([archive, newX]); allF = np.hstack([F, F_new])
        sel = np.argsort(allF)[:k]
        archive, F = allX[sel], allF[sel]
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(archive[ib].copy(), float(F[ib]), hist)

# --- GWO (Grey Wolf Optimizer) ---
def GWO_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    for t in range(gens):
        a = 2 - 2*(t/(gens-1))
        idx_sorted = np.argsort(F)
        alpha, beta, delta = X[idx_sorted[0]], X[idx_sorted[1]], X[idx_sorted[2]]
        for i in range(n):
            for leader in [alpha, beta, delta]:
                A = 2*a*rng.random(leader.size) - a
                C = 2*rng.random(leader.size)
                D = np.abs(C*leader - X[i])
                X[i] = leader - A*D
            X[i] /= 3.0
            X[i] = repair(X[i], cfg, idx)
        F = np.array([evaluator.fitness_only(x, idx) for x in X])
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- SCA (Sine Cosine Algorithm) ---
def SCA_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    best = X[np.argmin(F)].copy()
    for t in range(1, gens+1):
        r1 = 2 - 2*(t/(gens+1))
        r2 = 2*math.pi*rng.random(X.shape)
        r3 = 2*rng.random(X.shape) - 1
        r4 = rng.random(X.shape)
        for i in range(n):
            mask = r4[i] < 0.5
            X[i][mask] = X[i][mask] + r1*np.sin(r2[i][mask])*np.abs(r3[i][mask]*best[mask] - X[i][mask])
            X[i][~mask] = X[i][~mask] + r1*np.cos(r2[i][~mask])*np.abs(r3[i][~mask]*best[~mask] - X[i][~mask])
            X[i] = repair(X[i], cfg, idx)
            F[i] = evaluator.fitness_only(X[i], idx)
        if F.min() < evaluator.fitness_only(best, idx):
            best = X[np.argmin(F)].copy()
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# --- HHO (Harris Hawks Optimization) ---
def HHO_optimize(evaluator, cfg, idx, rng):
    n, gens = cfg.pop_size, cfg.n_generations
    X = np.vstack([random_solution(cfg, idx, rng) for _ in range(n)])
    F = np.array([evaluator.fitness_only(x, idx) for x in X])
    hist = [float(F.min())]
    best = X[np.argmin(F)].copy()
    for t in range(1, gens+1):
        E0 = 2*rng.random() - 1
        E = 2*E0*(1 - t/(gens+1))
        for i in range(n):
            if abs(E) >= 1:
                q = rng.random()
                if q < 0.5:
                    rand = random_solution(cfg, idx, rng)
                    X[i] = rand - rng.random(rand.size)*np.abs(rand - 2*rng.random(rand.size)*X[i])
                else:
                    j = rng.integers(0, n)
                    X[i] = (X[j] - X[i]) - rng.random(X[i].size)*(cfg.global_upper_bound*0.01)
            else:
                r = rng.random(X[i].size)
                X[i] = best - E*np.abs(2*r*best - X[i])
            X[i] = repair(X[i], cfg, idx); F[i] = evaluator.fitness_only(X[i], idx)
        if F.min() < evaluator.fitness_only(best, idx):
            best = X[np.argmin(F)].copy()
        hist.append(float(F.min()))
    ib = int(np.argmin(F))
    return wrap_return(X[ib].copy(), float(F[ib]), hist)

# ============================================================
# 8) Plots y exports
# ============================================================

def plot_membership_demo(cfg: ModelConfig, paths: Dict[str, pathlib.Path]):
    x = np.linspace(0, 100_000, 500)
    low = np.maximum(0, np.minimum((x-0)/(30_000-0), 1 - (x-0)/(30_000-0)))
    mid = np.maximum(0, np.minimum((x-20_000)/(50_000-20_000), 1 - (x-20_000)/(80_000-20_000)))
    high = np.maximum(0, np.minimum((x-50_000)/(80_000-50_000), 1 - (x-50_000)/(100_000-50_000)))
    a, m, b = 40_000, 60_000, 80_000
    centroid = triangular_centroid(a, m, b)
    plt.figure(figsize=(10,5))
    plt.plot(x, low, label="low")
    plt.plot(x, mid, label="medium")
    plt.plot(x, high, label="high")
    plt.axvline(centroid, linestyle="--", label=f"Centroid example = {centroid:,.0f}")
    plt.title("Triangular membership functions and centroid (demo)")
    plt.xlabel("Variable (units)"); plt.ylabel("Membership degree")
    plt.legend(); plt.tight_layout()
    savefig_all(paths["figs"]/ "membership_demo")

def plot_bars(df: pd.DataFrame, paths: Dict[str, pathlib.Path]):
    # Best Fitness
    plt.figure(figsize=(10,5))
    plt.bar(df["Algorithm"], df["Best Fitness"])
    plt.title("Best Fitness by Algorithm (lower is better)")
    plt.xlabel("Algorithm"); plt.ylabel("Best Fitness"); plt.tight_layout()
    savefig_all(paths["figs"]/ "best_fitness_bar")

    # Minimum cost
    plt.figure(figsize=(10,5))
    plt.bar(df["Algorithm"], df["Minimum Cost"])
    plt.title("Minimum cost of the best solution per algorithm")
    plt.xlabel("Algorithm"); plt.ylabel("Minimum cost"); plt.tight_layout()
    savefig_all(paths["figs"]/ "min_cost_bar")

    # penalty
    plt.figure(figsize=(10,5))
    plt.bar(df["Algorithm"], df["Penalty"])
    plt.title("Total penalty by algorithm")
    plt.xlabel("Algoritmo"); plt.ylabel("penalty"); plt.tight_layout()
    savefig_all(paths["figs"]/ "penalty_bar")

    # time
    plt.figure(figsize=(10,5))
    plt.bar(df["Algorithm"], df["Time (s)"])
    plt.title("runtime by algorithm")
    plt.xlabel("Algorithm"); plt.ylabel("time (s)"); plt.tight_layout()
    savefig_all(paths["figs"]/ "time_bar")

def plot_multiline_vectors(best_vectors: Dict[str, np.ndarray], paths: Dict[str, pathlib.Path]):
    plt.figure(figsize=(12,6))
    for name, vec in best_vectors.items():
        plt.plot(np.arange(vec.size), vec, label=name, linewidth=1)
    plt.title("optimal vectors by algorithm (multiline)")
    plt.xlabel("Variable index"); plt.ylabel("Assigned flow")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    savefig_all(paths["figs"]/ "optimal_vectors_multiline")

def plot_stacked_violations(violations_by_algo: Dict[str, Dict[str,int]], paths: Dict[str, pathlib.Path]):
    types = sorted({k for v in violations_by_algo.values() for k in v.keys()})
    algos = list(violations_by_algo.keys())
    M = np.zeros((len(algos), len(types)), dtype=int)
    for i, a in enumerate(algos):
        for j, t in enumerate(types):
            M[i, j] = violations_by_algo[a].get(t, 0)
    plt.figure(figsize=(12,6))
    bottom = np.zeros(len(algos))
    for j, t in enumerate(types):
        plt.bar(algos, M[:, j], bottom=bottom, label=t)
        bottom += M[:, j]
    plt.title("Violations by type (stacked bars)")
    plt.xlabel("Algorithm"); plt.ylabel("violations count")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    savefig_all(paths["figs"]/ "violations_stacked_bar")

def plot_convergence(histories: Dict[str, List[float]], paths: Dict[str, pathlib.Path]):
    # Combined
    plt.figure(figsize=(12,6))
    for name, h in histories.items():
        plt.plot(h, label=name, linewidth=1)
    plt.title("Convergence (best fitness per iteration)")
    plt.xlabel("Iteration"); plt.ylabel("Best Fitness"); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); savefig_all(paths["figs"]/ "convergence_all")
    # Individuals
    for name, h in histories.items():
        plt.figure(figsize=(8,4))
        plt.plot(h)
        plt.title(f"Convergence - {name}")
        plt.xlabel("Iteration"); plt.ylabel("Best Fitness"); plt.tight_layout()
        savefig_all(paths["figs"]/ f"Convergence - {name}")

def export_best_vectors_csv(best_vectors: Dict[str, np.ndarray], paths: Dict[str, pathlib.Path]):
    algos = []
    rows = []
    max_len = max(v.size for v in best_vectors.values())
    for name, vec in best_vectors.items():
        algos.append(name)
        r = np.zeros(max_len, dtype=float)
        r[:vec.size] = vec
        rows.append(r)
    cols = [f"var_{k+1}" for k in range(max_len)]
    dfv = pd.DataFrame(rows, index=algos, columns=cols).reset_index().rename(columns={"index":"Algorithm"})
    dfv.to_csv(paths["base"]/ "best_vectors.csv", index=False)

def export_violations_detail(violations_by_algo: Dict[str, Dict[str,int]], paths: Dict[str, pathlib.Path]):
    rows = []
    for algo, d in violations_by_algo.items():
        for k, v in d.items():
            rows.append({"Algorithm": algo, "Type": k, "Count": v})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Algorithm","Type","Count"])
    df.to_csv(paths["base"]/ "violations_detail.csv", index=False)

def export_histories(histories: Dict[str, List[float]], paths: Dict[str, pathlib.Path]):
    for name, h in histories.items():
        dfh = pd.DataFrame({"iteration": np.arange(len(h)), "best_fitness": h})
        dfh.to_csv(paths["logs"]/ f"{name}_history.csv", index=False)

# ============================================================
# 9) Benchmark orchestration
# ============================================================


def _plus__get_algorithm_map(include_hybrids=True):
    """Build the name→function mapping based on what is available in the module."""
    G = globals()
    base = {
        "GA": G.get("GA_optimize"), "PSO": G.get("PSO_optimize"), "DE": G.get("DE_optimize"),
        "SA": G.get("SA_optimize"), "HS": G.get("HS_optimize"), "ABC": G.get("ABC_optimize"),
        "VNS": G.get("VNS_optimize"), "TS": G.get("TS_optimize"), "ES": G.get("ES_optimize"),
        "EP": G.get("EP_optimize"), "FA": G.get("FA_optimize"), "CS": G.get("CS_optimize"),
        "WOA": G.get("WOA_optimize"), "BA": G.get("BA_optimize"), "TLBO": G.get("TLBO_optimize"),
        "MA": G.get("MA_optimize"), "GSA": G.get("GSA_optimize"), "ICA": G.get("ICA_optimize"),
        "BBO": G.get("BBO_optimize"), "ALO": G.get("ALO_optimize"), "QPSO": G.get("QPSO_optimize"),
        "DA": G.get("DA_optimize"), "ACOR": G.get("ACOR_optimize"), "GWO": G.get("GWO_optimize"),
        "SCA": G.get("SCA_optimize"), "HHO": G.get("HHO_optimize"),
    }
    if include_hybrids:
        base.update({
            "HYB_PSO_DE_GA": G.get("HYB_PSO_DE_GA_optimize"),
            "HYB_MEMETIC_TS_VNS": G.get("HYB_MEMETIC_TS_VNS_optimize"),
        })
    return {k:v for k,v in base.items() if v is not None}

def run_benchmark():
    cfg = ModelConfig()
    fix_all_seeds(cfg.base_seed)
    paths = ensure_dirs(cfg.results_dir)

    idx = Indexer(cfg.n_sources, cfg.n_sectors, cfg.n_periods)
    evaluator = ModelEvaluator(cfg)

    algos: Dict[str, Callable] = {
        "GA": GA_optimize, "PSO": PSO_optimize, "DE": DE_optimize, "SA": SA_optimize,
        "HS": HS_optimize, "ABC": ABC_optimize, "VNS": VNS_optimize, "TS": TS_optimize,
        "ES": ES_optimize, "EP": EP_optimize, "FA": FA_optimize, "CS": CS_optimize,
        "WOA": WOA_optimize, "BA": BA_optimize, "TLBO": TLBO_optimize, "MA": MA_optimize,
        "GSA": GSA_optimize, "ICA": ICA_optimize, "BBO": BBO_optimize, "ALO": ALO_optimize,
        "QPSO": QPSO_optimize, "DA": DA_optimize, "ACOR": ACOR_optimize, "GWO": GWO_optimize,
        "SCA": SCA_optimize, "HHO": HHO_optimize
    }
    rngs = make_rngs(cfg.base_seed, list(algos.keys()))

    rows = []
    best_vectors: Dict[str, np.ndarray] = {}
    violations_by_algo: Dict[str, Dict[str,int]] = {}
    histories: Dict[str, List[float]] = {}

    print("Running benchmark (maximum version)...\n")
    for name, fn in algos.items():
        t0 = time.time()
        best_x, best_f, hist = fn(evaluator, cfg, idx, rngs[name])
        dt = time.time() - t0
        er = evaluator.evaluate(best_x, idx)
        rows.append({
            "Algorithm": name,
            "Best Fitness": er.fitness,
            "Minimum Cost": er.cost,
            "Penalty": er.penalty,
            "Violations": er.n_violations,
            "Time (s)": dt
        })
        best_vectors[name] = best_x.copy()
        violations_by_algo[name] = er.violations_detail
        histories[name] = hist
        np.save(paths["base"]/ f"best_vector_{name}.npy", best_x)
        print(f"{name:>4s} | f*={er.fitness:,.2f} | cost={er.cost:,.2f} | pen={er.penalty:,.2f} | viol={er.n_violations} | t={dt:.2f}s")

    df = pd.DataFrame(rows).sort_values(by="Best Fitness", ascending=True).reset_index(drop=True)
    print("\Table sorted by Best Fitness (ascending):\n")
    print(df.to_string(index=False))
    df.to_csv(paths["base"]/ "benchmark_results.csv", index=False)

    # Exports y plots
    export_best_vectors_csv(best_vectors, paths)
    export_violations_detail(violations_by_algo, paths)
    export_histories(histories, paths)

    plot_membership_demo(cfg, paths)
    plot_bars(df, paths)
    plot_multiline_vectors(best_vectors, paths)
    plot_stacked_violations(violations_by_algo, paths)
    plot_convergence(histories, paths)

    # Final report
    best_name = df.loc[0, "Algorithm"]
    print(f"\nMejor algoritmo: {best_name}")
    print("Detalle de violations (por tipo) de la mejor solución:", violations_by_algo[best_name])
    print(f"\nArchivos generados en: {paths['base'].resolve()}")

if __name__ == "__wm_base__":  # desactivado en versión monolítica con PLUS
    run_benchmark()

# ======================================================================
# EXTENSION: 25 additional views (does not change anything from the above)
# They feed on artifacts already generated by run_benchmark()
# (CSV/NPY in results/, logs/). They save PNG/PDF/SVG in results/figs/.
# ======================================================================

# --- Local imports for extension ---
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from pathlib import Path as _Path

def _ext__safe_read_csv(p: _Path):
    try:
        return _pd.read_csv(p)
    except Exception as e:
        print(f"[NOTICE] Could not read {p}: {e}")
        return None

def _ext__infer_dims(n):
    # Heuristic for factoring n = I*J*T prioritizing small triples.
    candidates = []
    for I in range(2, 8):
        for J in range(2, 8):
            for T in range(2, 8):
                if I*J*T == n:
                    candidates.append((I,J,T))
    pref = [(3,3,2), (3,3,3), (4,3,2), (3,4,2), (4,4,2), (5,3,2)]
    for p in pref:
        if p in candidates:
            return p
    return candidates[0] if candidates else (n,1,1)

def _ext__load_artifacts(base_dir="results"):
    base = _Path(base_dir)
    figs = base / "figs"
    logs = base / "logs"
    figs.mkdir(parents=True, exist_ok=True)
    # Main table
    df = _ext__safe_read_csv(base / "benchmark_results.csv")
    if df is None:
        for alt in ["comparison_table.csv", "benchmark.csv", "tabla_comparativa.csv"]:
            if (base/alt).exists():
                df = _ext__safe_read_csv(base/alt)
                break
    if df is None:
        raise FileNotFoundError("No benchmark_results.csv found in 'results/'. Execute run benchmark() first.")

    # Optimal vectors
    best_vectors = {}
    for npy in sorted(base.glob("best_vector_*.npy")):
        name = npy.stem.replace("best_vector_","")
        try:
            best_vectors[name] = _np.load(npy)
        except Exception as e:
            print("[NOTICE] Error loading", npy, e)
    if not best_vectors and (base/"best_vectors.csv").exists():
        dfv = _pd.read_csv(base/"best_vectors.csv")
        alg_col = "Algorithm" if "Algorithm" in dfv.columns else dfv.columns[0]
        for _, row in dfv.iterrows():
            name = str(row[alg_col])
            vec = row.drop(labels=[alg_col]).values.astype(float)
            best_vectors[name] = vec

    # Stories of convergence
    histories = {}
    for f in sorted((logs).glob("*_history.csv")):
        name = f.stem.replace("_history","")
        try:
            d = _pd.read_csv(f)
            if "best_fitness" in d.columns:
                histories[name] = d["best_fitness"].tolist()
        except Exception as e:
            print("[WARNING] Error with history", f, e)

    # Detailed violations
    violations_detail = None
    if (base/"violations_detail.csv").exists():
        violations_detail = _pd.read_csv(base/"violations_detail.csv")
    else:
        if "Violations" in df.columns:
            violations_detail = _pd.DataFrame({
                "Algorithm": df["Algorithm"],
                "Type": ["Total"]*len(df),
                "Count": df["Violations"].values
            })

    return df, best_vectors, histories, violations_detail, base, figs

def _ext__savefig_all(figpath_noext: _Path):
    _plt.savefig(str(figpath_noext)+".png", dpi=150, bbox_inches="tight")
    _plt.savefig(str(figpath_noext)+".pdf", bbox_inches="tight")
    _plt.savefig(str(figpath_noext)+".svg", bbox_inches="tight")
    _plt.close()

def _ext__normalize_cols(df, cols):
    d = df.copy()
    for c in cols:
        v = d[c].values.astype(float)
        # For metrics to be minimized: invest in "higher is better" normalization
        if c.lower().startswith(("best", "min", "penalty", "viol", "time")):
            v = v.max() - v
        vmin, vmax = v.min(), v.max()
        d[c+"_norm"] = (v - vmin) / (vmax - vmin + 1e-12)
    return d

def generate_extra_visualizations_25():
    df, best_vectors, histories, vdetail, base, figs = _ext__load_artifacts()

    # 01) Correlation matrix
    cols = [c for c in ["Best Fitness","Minimum Cost","Penalty","Violations","Time (s)"] if c in df.columns]
    if len(cols) >= 2:
        C = df[cols].corr()
        _plt.figure(figsize=(6,5))
        _plt.imshow(C, interpolation="nearest")
        _plt.xticks(range(len(cols)), cols, rotation=45, ha="right"); _plt.yticks(range(len(cols)), cols)
        _plt.colorbar(label="Correlation")
        _plt.title("Metric correlation matrix")
        _ext__savefig_all(figs / "01_corr_metrics")

    # 02) Cost vs penalty
    if set(["Minimum Cost","Penalty"]).issubset(df.columns):
        _plt.figure(figsize=(7,5))
        _plt.scatter(df["Minimum Cost"], df["Penalty"])
        for _,r in df.iterrows():
            _plt.annotate(r["Algorithm"], (r["Minimum Cost"], r["Penalty"]), fontsize=8, xytext=(4,4), textcoords="offset points")
        _plt.xlabel("Minimum cost"); _plt.ylabel("penalty"); _plt.title("cost vs penalty")
        _ext__savefig_all(figs / "02_scatter_cost_penalty")

    # 03) Fitness vs cost
    if set(["Best Fitness","Minimum Cost"]).issubset(df.columns):
        _plt.figure(figsize=(7,5))
        sz = (df["Time (s)"] if "Time (s)" in df.columns else 1.0) * 10 + 30
        _plt.scatter(df["Minimum Cost"], df["Best Fitness"], s=sz)
        _plt.xlabel("Minimum cost"); _plt.ylabel("Best Fitness"); _plt.title("Fitness vs cost (tamaño ~ time)")
        _ext__savefig_all(figs / "03_scatter_fitness_cost")

    # 04) Violations vs penalty
    if set(["Violations","Penalty"]).issubset(df.columns):
        _plt.figure(figsize=(7,5))
        _plt.scatter(df["Violations"], df["Penalty"])
        _plt.xlabel("violations"); _plt.ylabel("penalty"); _plt.title("violations vs penalty")
        _ext__savefig_all(figs / "04_scatter_viol_penalty")

    # 05) Standardized radar (Top-5 por fitness)
    use = df.sort_values(by="Best Fitness", ascending=True).head(min(5, len(df))).reset_index(drop=True)
    radar_cols = [c for c in ["Best Fitness","Minimum Cost","Penalty","Violations","Time (s)"] if c in df.columns]
    if len(radar_cols) >= 3 and len(use) >= 2:
        dnorm = _ext__normalize_cols(use, radar_cols)
        labels = [c+"_norm" for c in radar_cols]
        angles = _np.linspace(0, 2*_np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        _plt.figure(figsize=(7,7))
        ax = _plt.subplot(111, polar=True)
        for _, r in dnorm.iterrows():
            data = r[labels].values.tolist()
            data += data[:1]
            ax.plot(angles, data, label=r["Algorithm"])
            ax.fill(angles, data, alpha=0.1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_cols)
        ax.set_yticklabels([])
        _plt.title("Normalized Metrics Radar (Top-5)"); _plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.1), fontsize=8)
        _ext__savefig_all(figs / "05_radar_norm")

    # 06) Boxplot of optimal vectors by algorithm
    if best_vectors:
        data = [v for _, v in sorted(best_vectors.items())]
        labels = [k for k,_ in sorted(best_vectors.items())]
        _plt.figure(figsize=(max(10, len(labels)*0.6),6))
        _plt.boxplot(data, labels=labels, showfliers=False)
        _plt.xticks(rotation=45, ha="right")
        _plt.ylabel("Value (assignment)"); _plt.title("Distribution of optimal vectors by algorithm")
        _ext__savefig_all(figs / "06_box_best_vectors_per_algo")

    # 07) Heatmap de optimal vectors
    if best_vectors:
        algos = list(best_vectors.keys())
        max_len = max(v.size for v in best_vectors.values())
        M = _np.zeros((len(algos), max_len))
        for i,a in enumerate(algos):
            v = best_vectors[a]
            M[i,:v.size] = v
        _plt.figure(figsize=(12, max(6, len(algos)*0.4)))
        _plt.imshow(M, aspect="auto", interpolation="nearest")
        _plt.colorbar(label="Valor")
        _plt.yticks(range(len(algos)), algos)
        _plt.xlabel("Variable index"); _plt.title("Optimal vectors heat map")
        _ext__savefig_all(figs / "07_heat_best_vectors")

    # 08) Variability by variable
    if best_vectors:
        max_len = max(v.size for v in best_vectors.values())
        M = []
        for a,v in best_vectors.items():
            row = _np.zeros(max_len); row[:v.size] = v
            M.append(row)
        M = _np.vstack(M)
        std = M.std(axis=0)
        _plt.figure(figsize=(12,4))
        _plt.bar(_np.arange(max_len), std)
        _plt.xlabel("Variable"); _plt.ylabel("Standard deviation"); _plt.title("Variability between algorithms by variable")
        _ext__savefig_all(figs / "08_var_importance_std")

    # 09) PCA de optimal vectors (2D)
    if best_vectors:
        A = []
        names = []
        max_len = max(v.size for v in best_vectors.values())
        for name, v in best_vectors.items():
            r = _np.zeros(max_len); r[:v.size]=v; A.append(r); names.append(name)
        X = _np.asarray(A)
        Xc = X - X.mean(axis=0, keepdims=True)
        U,S,Vt = _np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        _plt.figure(figsize=(7,6))
        _plt.scatter(Z[:,0], Z[:,1])
        for i,n in enumerate(names):
            _plt.annotate(n, (Z[i,0], Z[i,1]), fontsize=8, xytext=(4,4), textcoords="offset points")
        _plt.xlabel("PC1"); _plt.ylabel("PC2"); _plt.title("PCA de optimal vectors (algoritmos)")
        _ext__savefig_all(figs / "09_pca_best_vectors")

    # 10) K-means (k=3) over PCA
    if best_vectors:
        def kmeans(Z, k=3, iters=50, rng=_np.random.default_rng(0)):
            idx = rng.choice(len(Z), size=k, replace=False)
            C = Z[idx].copy()
            for _ in range(iters):
                D = ((Z[:,None,:]-C[None,:,:])**2).sum(axis=2)
                lab = D.argmin(axis=1)
                C_new = _np.vstack([Z[lab==j].mean(axis=0) if (lab==j).any() else C[j] for j in range(k)])
                if _np.allclose(C_new, C): break
                C = C_new
            return lab, C
        A = []
        names = []
        max_len = max(v.size for v in best_vectors.values())
        for name, v in best_vectors.items():
            r = _np.zeros(max_len); r[:v.size]=v; A.append(r); names.append(name)
        X = _np.asarray(A)
        Xc = X - X.mean(axis=0, keepdims=True)
        U,S,Vt = _np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :2] * S[:2]
        k = min(3, len(Z))
        lab, C = kmeans(Z, k=k)
        _plt.figure(figsize=(7,6))
        for j in range(k):
            idxs = _np.where(lab==j)[0]
            _plt.scatter(Z[idxs,0], Z[idxs,1], label=f"Cluster {j+1}")
        _plt.scatter(C[:,0], C[:,1], marker="x", s=100, label="Centroids")
        for i,n in enumerate(names):
            _plt.annotate(n, (Z[i,0], Z[i,1]), fontsize=8, xytext=(3,3), textcoords="offset points")
        _plt.legend(); _plt.title("Clustering (k-means) in PCA space")
        _ext__savefig_all(figs / "10_kmeans_pca")

    # 11) ECDF fitness algorithm
    if histories:
        _plt.figure(figsize=(8,6))
        for name,h in histories.items():
            x = _np.sort(_np.asarray(h))
            y = _np.linspace(0,1,len(x))
            _plt.plot(x,y,label=name)
        _plt.xlabel("Best Fitness"); _plt.ylabel("ECDF"); _plt.title("ECDF of fitness trajectories")
        _plt.legend(ncol=2, fontsize=8)
        _ext__savefig_all(figs / "11_ecdf_histories")

    # 12) Violin of trajectories
    if histories:
        data = [_np.asarray(h) for _,h in sorted(histories.items())]
        labels = [k for k,_ in sorted(histories.items())]
        _plt.figure(figsize=(max(10, len(labels)*0.6),6))
        _plt.violinplot(data, showmedians=True, showextrema=False)
        _plt.xticks(_np.arange(1,len(labels)+1), labels, rotation=45, ha="right")
        _plt.ylabel("Best fitness per iteration"); _plt.title("Convergence violin")
        _ext__savefig_all(figs / "12_violin_histories")

    # 13) Cumulative improvement step (best algorithm)
    if histories:
        best_name = sorted([(min(h), n) for n,h in histories.items()])[0][1]
        h = histories[best_name]
        best_so_far = _np.minimum.accumulate(h)
        _plt.figure(figsize=(8,4))
        _plt.step(_np.arange(len(h)), best_so_far, where="post")
        _plt.xlabel("Iteration"); _plt.ylabel("Best-so-far"); _plt.title(f"Cumulative improvement - {best_name}")
        _ext__savefig_all(figs / "13_step_best_algo")

    # 14) Area (mobile medium) Top-3
    if histories:
        K = min(3, len(histories))
        top = sorted(histories.items(), key=lambda kv: min(kv[1]))[:K]
        _plt.figure(figsize=(9,5))
        for name,h in top:
            h = _np.asarray(h)
            w = 5
            ma = _np.convolve(h, _np.ones(w)/w, mode="valid")
            _plt.fill_between(_np.arange(ma.size), ma, alpha=0.2, label=name)
            _plt.plot(ma)
        _plt.xlabel("Iteration"); _plt.ylabel("mobile fitness medium"); _plt.title("Convergence smoothing (Top-3)")
        _plt.legend()
        _ext__savefig_all(figs / "14_area_ma_top3")

    # 15) Histogram of improvements of the best algorithm
    if histories:
        best_name = sorted([(min(h), n) for n,h in histories.items()])[0][1]
        h = _np.asarray(histories[best_name])
        delta = _np.diff(h)
        imp = delta[delta<0]
        if imp.size>0:
            _plt.figure(figsize=(7,4))
            _plt.hist(-imp, bins=20)
            _plt.xlabel("Magnitude of improvement"); _plt.ylabel("Frequency"); _plt.title(f"Distribution of improvements - {best_name}")
            _ext__savefig_all(figs / "15_hist_improvements_best")

    # 16) Iterations to reach 105% of the optimum
    if histories:
        names = []; its = []
        for name,h in histories.items():
            h = _np.asarray(h)
            target = h.min()*1.05
            hit = _np.where(h <= target)[0]
            its.append(int(hit[0]) if hit.size else len(h))
            names.append(name)
        _plt.figure(figsize=(10,5))
        _plt.bar(names, its)
        _plt.xticks(rotation=45, ha="right")
        _plt.ylabel("Iterations"); _plt.title("Convergence speed (at 105% of optimum)")
        _ext__savefig_all(figs / "16_iters_to_105pct")

    # 17) Bubble: cost vs penalty (size ~ Violations)
    if set(["Minimum Cost","Penalty","Best Fitness"]).issubset(df.columns):
        s = (df["Violations"] if "Violations" in df.columns else 1.0)*20 + 40
        _plt.figure(figsize=(8,6))
        _plt.scatter(df["Minimum Cost"], df["Penalty"], s=s)
        for _,r in df.iterrows():
            _plt.annotate(r["Algorithm"], (r["Minimum Cost"], r["Penalty"]), fontsize=8, xytext=(3,3), textcoords="offset points")
        _plt.xlabel("cost"); _plt.ylabel("penalty"); _plt.title("Bubble: cost vs penalty (size~violations)")
        _ext__savefig_all(figs / "17_bubble_cost_penalty_viol")

    # 18) Embedded table sorted by fitness
    dft = df.sort_values("Best Fitness", ascending=True).reset_index(drop=True) if "Best Fitness" in df.columns else df.copy()
    fig, ax = _plt.subplots(figsize=(10, 0.5 + 0.35*len(dft)))
    ax.axis('off')
    tbl = ax.table(cellText=dft.values, colLabels=dft.columns, loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
    ax.set_title("Comparative table (sorted by Best Fitness)" if "Best Fitness" in df.columns else "Comparative table")
    _ext__savefig_all(figs / "18_table_comparison")

    # 19) Sankey sources → sectors (best algorithm)
    if best_vectors and len(best_vectors)>0:
        best_algo = dft.iloc[0]["Algorithm"] if "Algorithm" in dft.columns else list(best_vectors.keys())[0]
        v = best_vectors.get(best_algo, next(iter(best_vectors.values())))
        I,J,T = _ext__infer_dims(len(v))
        X = v.reshape(I,J,T, order="F")
        flows = X.sum(axis=2)  # (I,J)
        try:
            from matplotlib.sankey import Sankey as _Sankey
            S = _Sankey(unit=None, format='%.0f')
            in_flows = flows.sum(axis=1)
            out_flows = flows.sum(axis=0)
            total_in = in_flows.sum(); total_out = out_flows.sum()
            if abs(total_in - total_out) > 1e-6:
                scale = 0.5*(total_in+total_out)/max(total_in,total_out)
                in_flows = in_flows*scale; out_flows = out_flows*scale
            flows_list = list(in_flows.tolist()) + list((-out_flows).tolist())
            labels = [f"S{i+1}" for i in range(I)] + [f"Sec{j+1}" for j in range(J)]
            S.add(flows=flows_list, labels=labels, orientations=[1]*I+[-1]*J)
            S.finish()
            _plt.title(f"Sankey sources → sectors (algorithm {best_algo})")
            _ext__savefig_all(figs / "19_sankey_sources_to_sectors")
        except Exception as e:
            print("[NOTICE] Sankey not available:", e)

    # 20) Stacked area by sector over time (best algorithm)
    if best_vectors and len(best_vectors)>0:
        best_algo = dft.iloc[0]["Algorithm"] if "Algorithm" in dft.columns else list(best_vectors.keys())[0]
        v = best_vectors.get(best_algo, next(iter(best_vectors.values())))
        I,J,T = _ext__infer_dims(len(v))
        X = v.reshape(I,J,T, order="F")
        per_sector = X.sum(axis=0)  # (J,T)
        _plt.figure(figsize=(9,5))
        x = _np.arange(1, T+1)
        bottom = _np.zeros(T)
        for j in range(J):
            _plt.bar(x, per_sector[j,:], bottom=bottom, label=f"sector {j+1}")
            bottom += per_sector[j,:]
        _plt.xlabel("Period"); _plt.ylabel("Flujo"); _plt.title("Composition by sector (stacked in periods)")
        _plt.legend(ncol=2, fontsize=8)
        _ext__savefig_all(figs / "20_stacked_by_sector_over_time")

    # 21) Histogram of values ​​of the best vector
    if best_vectors and len(best_vectors)>0:
        best_algo = dft.iloc[0]["Algorithm"] if "Algorithm" in dft.columns else list(best_vectors.keys())[0]
        v = best_vectors.get(best_algo, next(iter(best_vectors.values())))
        _plt.figure(figsize=(7,4))
        _plt.hist(v, bins=20)
        _plt.xlabel("Valor"); _plt.ylabel("Frequency"); _plt.title(f"Distribution of assignments - {best_algo}")
        _ext__savefig_all(figs / "21_hist_best_vector")

    # 22) CDF of the best vector
    if best_vectors and len(best_vectors)>0:
        best_algo = dft.iloc[0]["Algorithm"] if "Algorithm" in dft.columns else list(best_vectors.keys())[0]
        v = best_vectors.get(best_algo, next(iter(best_vectors.values())))
        x = _np.sort(v); y = _np.linspace(0,1,len(x))
        _plt.figure(figsize=(7,4))
        _plt.plot(x,y)
        _plt.xlabel("Valor"); _plt.ylabel("CDF"); _plt.title("CDF of assignments (best vector)")
        _ext__savefig_all(figs / "22_cdf_best_vector")

    # 23) Top 10 largest assignments
    if best_vectors and len(best_vectors)>0:
        best_algo = dft.iloc[0]["Algorithm"] if "Algorithm" in dft.columns else list(best_vectors.keys())[0]
        v = best_vectors.get(best_algo, next(iter(best_vectors.values())))
        idx = _np.argsort(v)[-10:][::-1]
        _plt.figure(figsize=(8,4))
        _plt.bar([str(i) for i in idx], v[idx])
        _plt.xlabel("Variable index"); _plt.ylabel("Valor"); _plt.title("Top-10 Best Vector Assignments")
        _ext__savefig_all(figs / "23_top10_vars_best_vector")

    # 24) Heatmap of violations by type and algorithm
    if vdetail is not None and len(vdetail)>0:
        types = sorted(vdetail["Type"].unique().tolist())
        algos = sorted(vdetail["Algorithm"].unique().tolist())
        M = _np.zeros((len(algos), len(types)))
        for i,a in enumerate(algos):
            sub = vdetail[vdetail["Algorithm"]==a]
            for j,t in enumerate(types):
                val = sub[sub["Type"]==t]["Count"].sum()
                M[i,j] = val
        _plt.figure(figsize=(12, max(6, len(algos)*0.4)))
        _plt.imshow(M, aspect="auto", interpolation="nearest")
        _plt.colorbar(label="withteo")
        _plt.yticks(range(len(algos)), algos); _plt.xticks(range(len(types)), types, rotation=45, ha="right")
        _plt.title("violations by type and algorithm (heatmap)")
        _ext__savefig_all(figs / "24_heat_violations_type_algo")

    # 25) Approximate Pareto envelope: cost–penalty
    if set(["Minimum Cost","Penalty"]).issubset(df.columns):
        D = df[["Minimum Cost","Penalty","Algorithm"]].copy().sort_values(["Minimum Cost","Penalty"])
        env_x, env_y = [], []
        best_y = float("inf")
        for _,r in D.iterrows():
            if r["Penalty"] < best_y:
                env_x.append(r["Minimum Cost"]); env_y.append(r["Penalty"]); best_y = r["Penalty"]
        _plt.figure(figsize=(7,5))
        _plt.scatter(D["Minimum Cost"], D["Penalty"])
        _plt.plot(env_x, env_y, linestyle="--")
        for _,r in D.iterrows():
            _plt.annotate(r["Algorithm"], (r["Minimum Cost"], r["Penalty"]), fontsize=7, xytext=(2,2), textcoords="offset points")
        _plt.xlabel("cost"); _plt.ylabel("penalty"); _plt.title("Approximate front cost-penalty")
        _ext__savefig_all(figs / "25_pareto_envelope")

    print(f"[OK] 25 new views were saved in: {figs.resolve()}")

# --- Automatic shooting after run_benchmark() ---
if __name__ == "__wm_base__":  # Disabled in monolithic version with PLUS
    try:
        generate_extra_visualizations_25()
    except Exception as e:
        print("[NOTICE] It was not possible to generate the 25 views:", e)



# ======================================================================
# ADDITIONAL MODULE: Implementation of the recommended Plan 3.2
# (All of the above is kept in full; this block ONLY adds)
# ----------------------------------------------------------------------
# withtenido:
#  A) Indicador AHP (combinación multi-criterio)
#  B) Frente de Pareto y cálculo de Hipervolumen (HV)
#  C) Portafolio de híbridos secuenciales:
#       - HYB_PSO_DE_GA_optimize
#       - HYB_MEMETIC_TS_VNS_optimize
#  D) Narrative Scenarios (Normal, Moderate Drought, Severe Drought,
#     Budget cuts, rising demand, quality shock) and run
#     benchmark by scenario for the best algorithms of the base benchmark.
#  E) Feasibility traceability: matrix (period × type) of violations
#     and timeline of % of constraints satisfied.
#  F) Wrappers de adaptive penalty + Comparador Feasible-First
#     (without altering the base algorithms).
#  G) Metadatos de referencias (README lógico de trazabilidad científica).
# ----------------------------------------------------------------------
# Notes:
#  - No previous functionality is modified; existing utilities are reused.
#  - Artifacts generated by run_benchmark() (CSV, NPY) are used where appropriate
#  - To limit execution times in scenarios, Top-K is filtered by fitness.
# ======================================================================

# --- Imports from the environment already present in the main script ---
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt
from collections import defaultdict as _defaultdict, deque as _deque
from dataclasses import replace as _dc_replace
import math as _math
import time as _time
from pathlib import Path  # Importation required for routes

# ----------------------------------------------------------------------
# Local utilities
# ----------------------------------------------------------------------
def _plus__norm01_invert_minimization(values):
    """Normalizes to [0,1] as 'higher is better' when the original metric is minimization."""
    v = _np.asarray(values, dtype=float)
    vmax, vmin = float(v.max()), float(v.min())
    if _math.isclose(vmax, vmin):
        return _np.zeros_like(v, dtype=float)
    inv = vmax - v  # bigger => better
    return (inv - inv.min()) / (inv.max() - inv.min())

def _plus__ensure_dirs(base):
    base.mkdir(parents=True, exist_ok=True)
    figs = base / "figs_ext"
    figs.mkdir(parents=True, exist_ok=True)
    return {"base": base, "figs": figs}

def _plus__savefig(figpath_noext):
    _plt.savefig(str(figpath_noext)+".png", dpi=150, bbox_inches="tight")
    _plt.savefig(str(figpath_noext)+".pdf", bbox_inches="tight")
    _plt.savefig(str(figpath_noext)+".svg", bbox_inches="tight")
    _plt.close()

def _plus__get_algos_dict():
    # Mapping names to functions (includes the previous 26 + new hybrids)
    algos = {
        "GA": GA_optimize, "PSO": PSO_optimize, "DE": DE_optimize, "SA": SA_optimize,
        "HS": HS_optimize, "ABC": ABC_optimize, "VNS": VNS_optimize, "TS": TS_optimize,
        "ES": ES_optimize, "EP": EP_optimize, "FA": FA_optimize, "CS": CS_optimize,
        "WOA": WOA_optimize, "BA": BA_optimize, "TLBO": TLBO_optimize, "MA": MA_optimize,
        "GSA": GSA_optimize, "ICA": ICA_optimize, "BBO": BBO_optimize, "ALO": ALO_optimize,
        "QPSO": QPSO_optimize, "DA": DA_optimize, "ACOR": ACOR_optimize, "GWO": GWO_optimize,
        "SCA": SCA_optimize, "HHO": HHO_optimize,
        # Hybrids added in this module:
        "HYB_PSO_DE_GA": HYB_PSO_DE_GA_optimize,
        "HYB_MEMETIC_TS_VNS": HYB_MEMETIC_TS_VNS_optimize,
    }
    return algos

# ----------------------------------------------------------------------
# A) AHP indicator
# ----------------------------------------------------------------------
def _plus__ahp_weights_from_pairwise(M):
    """Calculates AHP weights from a pairwise (nxn) comparison matrix."""
    w, v = _np.linalg.eig(_np.asarray(M, dtype=float))
    idx = int(_np.argmax(w.real))
    vec = _np.abs(v[:, idx].real)
    vec = vec / vec.sum()
    return vec

def _plus__ahp_score(df, cols, pairwise=None, weights=None, invert_minimization=True):
    """Returns AHP score for df[cols]."""
    X = _np.column_stack([
        _plus__norm01_invert_minimization(df[c].values) if invert_minimization else df[c].values
        for c in cols
    ])
    if pairwise is not None:
        w = _plus__ahp_weights_from_pairwise(pairwise)
    else:
        w = _np.asarray(weights if weights is not None else [1.0/len(cols)]*len(cols), dtype=float)
        w = w / w.sum()
    score = X @ w
    return score, w

# ----------------------------------------------------------------------
# B) Pareto and hypervolume
# ----------------------------------------------------------------------
def _plus__pareto_mask(points):
    """Pareto mask (True if not dominated) minimizing across all objectives."""
    P = _np.asarray(points, dtype=float)
    n = P.shape[0]
    mask = _np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if _np.all(P[j] <= P[i]) and _np.any(P[j] < P[i]):
                mask[i] = False
                break
    return mask

def _plus__hypervolume(points, ref):
    """2D Hypervolume for Minimization (approx. per sweep)."""
    P = _np.asarray(points, dtype=float)
    mask = _plus__pareto_mask(P)
    F = P[mask]
    if F.size == 0:
        return 0.0, F
    F = F[_np.argsort(F[:,0])]
    hv = 0.0
    prev_cost = ref[0]
    prev_pen  = ref[1]
    for cost, pen in F[::-1]:
        width  = prev_cost - cost
        height = max(0.0, ref[1] - pen)
        hv += max(0.0, width) * max(0.0, height)
        prev_cost = cost
        prev_pen  = min(prev_pen, pen)
    return float(hv), F

# ----------------------------------------------------------------------
# F) Wrappers: adaptive penalty + Feasible-First
# ----------------------------------------------------------------------
class FeasibleFirstAdaptiveEvaluator:
    """
    ModelEvaluator wrapper that:
    - Applies an ADAPTIVE penalty factor (alpha) on top of the raw penalty of the base evaluator.
    - Returns a fitness SCALAR with FEASIBLE-FIRST policy for selection operators.
    - It doesn't modify the base evaluator or the algorithms; it just intercepts fitness_only(...) and delegates the rest.
    """
    def __init__(self,
                 base_evaluator,
                 big_M=1e12,
                 alpha_init=1.0,
                 alpha_min=0.25,
                 alpha_max=16.0,
                 window=200,
                 up_ratio=0.60,
                 down_ratio=0.20,
                 step_up=1.10,
                 step_down=0.95):
        self.base = base_evaluator
        self.big_M = float(big_M)
        self.alpha = float(alpha_init)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.window = int(window)
        self.up_ratio = float(up_ratio)
        self.down_ratio = float(down_ratio)
        self.step_up = float(step_up)
        self.step_down = float(step_down)
        self._feas_hist = _deque(maxlen=self.window)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _update_alpha(self, feasible_flag):
        self._feas_hist.append(bool(feasible_flag))
        if len(self._feas_hist) < max(10, self.window//4):
            return
        feas_rate = sum(self._feas_hist) / len(self._feas_hist)
        viol_rate = 1.0 - feas_rate
        if viol_rate > self.up_ratio:
            self.alpha = min(self.alpha_max, self.alpha * self.step_up)
        elif viol_rate < self.down_ratio:
            self.alpha = max(self.alpha_min, self.alpha * self.step_down)

    def _wrapped_scalar(self, er):
        pen_raw = float(er.penalty)
        feas = (pen_raw <= 1e-12)
        self._update_alpha(feas)
        if feas:
            return float(er.fitness)
        pen_adapt = self.alpha * pen_raw
        return self.big_M * (1.0 + pen_adapt) + float(er.fitness)

    def evaluate(self, x, idx):
        return self.base.evaluate(x, idx)

    def fitness_only(self, x, idx):
        er = self.base.evaluate(x, idx)
        return self._wrapped_scalar(er)

# ----------------------------------------------------------------------
# C) Sequential hybrids (added)
# ----------------------------------------------------------------------
def _plus__clone_cfg(cfg, **kw):
    return _dc_replace(cfg, **kw)

def _plus__local_DE_around(evaluator, cfg, idx, rng, x0, pop=20, gens=30, F=0.6, CR=0.7, radius=0.05):
    ub = cfg.global_upper_bound
    P = _np.vstack([repair(x0 + rng.normal(0.0, radius*ub, size=x0.size), cfg, idx) for _ in range(pop)])
    Fvals = _np.array([evaluator.fitness_only(x, idx) for x in P])
    best = P[int(_np.argmin(Fvals))].copy(); fbest = float(Fvals.min()); hist=[fbest]
    for _ in range(gens):
        for i in range(pop):
            a,b,c = rng.integers(0, pop, size=3)
            while len({a,b,c,i})<4:
                a,b,c = rng.integers(0, pop, size=3)
            mutant = P[a] + F*(P[b]-P[c])
            cross  = rng.random(x0.size) < CR
            trial  = _np.where(cross, mutant, P[i])
            trial  = repair(trial, cfg, idx)
            ft = evaluator.fitness_only(trial, idx)
            if ft < Fvals[i]:
                P[i] = trial; Fvals[i] = ft
                if ft < fbest:
                    best, fbest = trial.copy(), float(ft)
        hist.append(fbest)
    return best, fbest, hist

def _plus__local_TS(evaluator, cfg, idx, rng, x0, iters=200, radius=0.03, tabu_size=25):
    best = repair(x0.copy(), cfg, idx)
    fbest = evaluator.fitness_only(best, idx)
    hist=[fbest]
    tabu = []
    for _ in range(iters):
        cand = repair(best + rng.normal(0.0, radius*cfg.global_upper_bound, size=best.size), cfg, idx)
        key  = tuple(_np.round(cand,5).tolist())
        if key in tabu:
            continue
        f = evaluator.fitness_only(cand, idx)
        if f < fbest:
            best, fbest = cand, f
        tabu.append(key)
        if len(tabu) > tabu_size:
            tabu.pop(0)
        hist.append(fbest)
    return best, fbest, hist

def _plus__local_VNS(evaluator, cfg, idx, rng, x0, kmax=5, iters_per_k=40, base_radius=0.02):
    best = repair(x0.copy(), cfg, idx)
    fbest = evaluator.fitness_only(best, idx); hist=[fbest]
    for k in range(1, kmax+1):
        radius = base_radius * k
        improved = False
        for _ in range(iters_per_k):
            cand = repair(best + rng.normal(0.0, radius*cfg.global_upper_bound, size=best.size), cfg, idx)
            f = evaluator.fitness_only(cand, idx)
            if f < fbest:
                best, fbest = cand, f
                improved = True
        hist.append(fbest)
        if not improved:
            continue
    return best, fbest, hist

def HYB_PSO_DE_GA_optimize(evaluator, cfg, idx, rng):
    """3-stage hybrid: PSO (exploration) → Local DE (intensification) → Reduced GA (refinement)."""
    cfg1 = _plus__clone_cfg(cfg, n_generations=max(20, cfg.n_generations//3), pop_size=max(30, cfg.pop_size//2))
    x1, f1, h1 = PSO_optimize(evaluator, cfg1, idx, rng)
    x2, f2, h2 = _plus__local_DE_around(evaluator, cfg, idx, rng, x1, pop=24, gens=max(30, cfg.n_generations//4))
    cfg3 = _plus__clone_cfg(cfg, n_generations=max(20, cfg.n_generations//4), pop_size=max(24, cfg.pop_size//2))
    x3, f3, h3 = GA_optimize(evaluator, cfg3, idx, rng)
    bestx, bestf = (x1, f1)
    if f2 < bestf: bestx, bestf = (x2, f2)
    if f3 < bestf: bestx, bestf = (x3, f3)
    hist = list(h1) + list(h2) + list(h3)
    return bestx, float(bestf), hist

def HYB_MEMETIC_TS_VNS_optimize(evaluator, cfg, idx, rng):
    """Memetic hybrid + local search: reduced MA → TS → VNS."""
    cfg1 = _plus__clone_cfg(cfg, n_generations=max(30, cfg.n_generations//3), pop_size=max(30, cfg.pop_size//2))
    x1, f1, h1 = MA_optimize(evaluator, cfg1, idx, rng)
    x2, f2, h2 = _plus__local_TS(evaluator, cfg, idx, rng, x1, iters=max(150, cfg.n_generations))
    x3, f3, h3 = _plus__local_VNS(evaluator, cfg, idx, rng, x2, kmax=5, iters_per_k=40)
    bestx, bestf = (x1, f1)
    if f2 < bestf: bestx, bestf = (x2, f2)
    if f3 < bestf: bestx, bestf = (x3, f3)
    hist = list(h1) + list(h2) + list(h3)
    return bestx, float(bestf), hist

# ----------------------------------------------------------------------
# D) Narrative scenarios and benchmark by scenario
# ----------------------------------------------------------------------
class ScenarioModelEvaluator(ModelEvaluator):
    """Extends the evaluator for demand/budget/availability/quality multipliers."""
    def __init__(self, cfg, demand_mult=1.0, budget_mult=1.0, avail_mult=1.0, micro_mult=1.0):
        super().__init__(cfg)
        self.data["demand"] = self.data["demand"] * float(demand_mult)
        self.data["budget_max"] = self.data["budget_max"] * float(budget_mult)
        self.data["avail"] = self.data["avail"] * float(avail_mult)
        self.data["micro_withc_source"] = self.data["micro_withc_source"] * float(micro_mult)

def _plus__scenario_defs():
    return [
        {"name":"Normal", "ov":{}, "mult":{"demand":1.00, "budget":1.00, "avail":1.00, "micro":1.00}},
        {"name":"Sequia_moderada", "ov":{"env_frac":0.85}, "mult":{"demand":1.00, "budget":1.00, "avail":0.90, "micro":1.00}},
        {"name":"Sequia_severa", "ov":{"env_frac":0.80, "drought_resilience_frac":0.95}, "mult":{"demand":1.00, "budget":1.00, "avail":0.80, "micro":1.00}},
        {"name":"Recorte_presup", "ov":{}, "mult":{"demand":1.00, "budget":0.85, "avail":1.00, "micro":1.00}},
        {"name":"Alza_demand", "ov":{}, "mult":{"demand":1.05, "budget":1.00, "avail":1.00, "micro":1.00}},
        {"name":"Shock_calidad", "ov":{}, "mult":{"demand":1.00, "budget":1.00, "avail":1.00, "micro":1.20}},
    ]

def _plus__select_top_algorithms(results_csv, k=6):
    df = _pd.read_csv(results_csv)
    df = df.sort_values(by=["Best Fitness","Penalty","Minimum Cost","Time (s)"], ascending=[True, True, True, True])
    return df["Algorithm"].head(int(k)).tolist()

def _plus__run_scenario(name, ov, mult, alg_names=None,
                        use_wrappers=True,
                        wrapper_kwargs=None):
    """
    Runs one benchmark per scenario. If use_wrappers=True, wraps the evaluator in a
    FeasibleFirstAdaptiveEvaluator to impose feasible-first and adaptive penalties.
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = dict(big_M=1e12, alpha_init=1.0, alpha_min=0.25, alpha_max=16.0,
                              window=200, up_ratio=0.60, down_ratio=0.20, step_up=1.10, step_down=0.95)

    cfg = ModelConfig()
    for key, val in ov.items():
        setattr(cfg, key, val)
    fix_all_seeds(cfg.base_seed)
    idx = Indexer(cfg.n_sources, cfg.n_sectors, cfg.n_periods)

    base_eval = ScenarioModelEvaluator(cfg,
                                       demand_mult=mult.get("demand",1.0),
                                       budget_mult=mult.get("budget",1.0),
                                       avail_mult=mult.get("avail",1.0),
                                       micro_mult=mult.get("micro",1.0))
    evaluator = (FeasibleFirstAdaptiveEvaluator(base_eval, **wrapper_kwargs)
                 if use_wrappers else base_eval)

    paths = _plus__ensure_dirs(cfg.results_dir / "scenarios" / name)
    algos = _plus__get_algos_dict()
    if alg_names is None:
        alg_names = list(algos.keys())
    rngs = make_rngs(cfg.base_seed, alg_names)

    rows = []
    best_vectors = {}
    violations_by_algo = {}
    histories = {}

    for an in alg_names:
        fn = algos[an]
        t0 = _time.time()
        best_x, best_f, hist = fn(evaluator, cfg, idx, rngs[an])
        dt = _time.time() - t0
        er = evaluator.evaluate(best_x, idx)
        rows.append({"Algorithm": an, "Best Fitness": er.fitness, "Minimum Cost": er.cost,
                     "Penalty": er.penalty, "Violations": er.n_violations, "Time (s)": dt})
        best_vectors[an] = best_x.copy()
        violations_by_algo[an] = er.violations_detail
        histories[an] = hist
        _np.save(paths["base"]/ f"best_vector_{an}.npy", best_x)
        _pd.DataFrame({"Fitness": hist}).to_csv(paths["base"]/ f"{an}_history.csv", index=False)

    dfr = _pd.DataFrame(rows).sort_values(by=["Best Fitness","Penalty","Minimum Cost","Time (s)"])
    dfr.to_csv(paths["base"]/ "benchmark_results.csv", index=False)

    rowsv=[]
    for an, vd in violations_by_algo.items():
        for key, val in vd.items():
            rowsv.append({"Algorithm": an, "Type": key, "Count": val})
    _pd.DataFrame(rowsv).to_csv(paths["base"]/ "violations_detail.csv", index=False)

    # Pareto cost-penalty by scenario
    pts = dfr[["Minimum Cost","Penalty"]].to_numpy()
    mask = _plus__pareto_mask(pts)
    _plt.figure(figsize=(7,6))
    _plt.scatter(pts[~mask,0], pts[~mask,1], alpha=0.5, label="Dominated")
    _plt.scatter(pts[mask,0], pts[mask,1], label="Not dominated")
    _plt.xlabel("cost"); _plt.ylabel("penalty"); _plt.title(f"Pareto cost-penalty • Scenario: {name}")
    _plt.legend()
    _plus__savefig(paths["figs"]/ "pareto_cost_penalty")

    return dfr, best_vectors, violations_by_algo, paths

def _plus__compute_reliability_resilience(summary_per_scn):
    """Reliability = % scenarios with penalty 0; Resilience = proxy with penalty in drought."""
    algos = sorted({an for scn, dfr in summary_per_scn.items() for an in dfr["Algorithm"]})
    rows=[]
    for an in algos:
        penalties = []
        drought_penalties=[]
        for scn, dfr in summary_per_scn.items():
            p = float(dfr.loc[dfr["Algorithm"]==an, "Penalty"].values[0])
            penalties.append(p)
            if "Sequia" in scn:
                drought_penalties.append(p)
        reliability = float(_np.mean([1.0 if p<=1e-9 else 0.0 for p in penalties]))
        if len(drought_penalties)==0:
            resilience = 1.0
        else:
            dp = _np.asarray(drought_penalties, dtype=float)
            resilience = 1.0 - (dp.mean() / (dp.max()+1e-9))
            resilience = float(max(0.0, min(1.0, resilience)))
        rows.append({"Algorithm": an, "Reliability": reliability, "Resilience": resilience})
    return _pd.DataFrame(rows)

# ----------------------------------------------------------------------
# E) Feasibility traceability by period
# ----------------------------------------------------------------------
def _plus__violations_matrix_by_period(evaluator, cfg, idx, x):
    """Generates a matrix (t × type) of violations (0/1 per check) and % satisfied."""
    D = evaluator.data
    X = idx.to_tensor(x)  # (I,J,T)

    types = ["ambiental","demand_min","cap_total","cap_pre","cap_primary","cap_sewithdary","cap_tertiary",
             "drought","strategic_reserve","Budget","Infrastructure","Micropollutants",
             "Monitoring","Maintenance","education"]
    rows = []
    perc_ok = []

    for t in range(idx.T):
        counts = _defaultdict(int)
        checks = 0
        # Environmental by source
        for i in range(idx.I):
            used = _np.sum(X[i, :, t])
            max_i = cfg.env_frac * D["avail"][t, i]
            checks += 1
            if used > max_i: counts["ambiental"] += 1

        # minimum demand by sector
        for j in range(idx.J):
            supplied = _np.sum(X[:, j, t]); demand_min = cfg.demand_coverage_min * D["demand"][t, j]
            checks += 1
            if supplied < demand_min: counts["demand_min"] += 1

        # total capacity and substages if they exist
        if "treat_cap_total" in D:
            checks += 1
            tot = _np.sum(X[:, :, t])
            if tot > D["treat_cap_total"]: counts["cap_total"] += 1
        for key, name in [("treat_cap_pre","cap_pre"),("treat_cap_primary","cap_primary"),
                          ("treat_cap_sewithdary","cap_sewithdary"),("treat_cap_tertiary","cap_tertiary")]:
            if key in D:
                checks += 1
                if _np.sum(X[:, :, t]) > D[key]: counts[name] += 1

        # Drought (resilience)
        checks += 1
        if _np.sum(X[:, :, t]) < (cfg.drought_resilience_frac * _np.sum(D["demand"][t, :])):
            counts["sequía"] += 1

        # Strategic reserve by sector
        for j in range(idx.J):
            checks += 1
            if _np.sum(X[:, j, t]) < D["strategic_min"][t, j]:
                counts["strategic_reserve"] += 1

        # Budget
        checks += 1
        cost_t = 0.0
        for i in range(idx.I):
            for j in range(idx.J):
                unit = D["cost"][i] + D["trans_cost"][i, j]
                cost_t += unit * _np.sum(X[i, j, t])
        if cost_t > D["budget_max"][t]: counts["Budget"] += 1

        # Infrastructure
        checks += 1
        infra_load = 0.0
        for i in range(idx.I):
            for j in range(idx.J):
                infra_load += D["trans_coef"][i, j] * _np.sum(X[i, j, t])
        if infra_load > D["infra_cap"][t]: counts["Infrastructure"] += 1

        # Micropollutants
        for j in range(idx.J):
            checks += 1
            denom = _np.sum(X[:, j, t])
            if denom > 0:
                numer = _np.sum(D["micro_withc_source"] * X[:, j, t])
                c_j = numer / denom
                if c_j > D["micro_max_sector"][j]:
                    counts["Micropollutants"] += 1

        # Monitoring, maintenance, education
        checks += 1
        mon = 0.0
        for i in range(idx.I):
            for j in range(idx.J):
                mon += D["qmon_weight"][i, j] * _np.sum(X[i, j, t])
        if mon < D["qmon_min"][t]: counts["Monitoring"] += 1

        checks += 1
        mload = 0.0
        for i in range(idx.I):
            for j in range(idx.J):
                mload += D["maint_weight"][i, j] * _np.sum(X[i, j, t])
        if mload < D["maint_min"][t]: counts["Maintenance"] += 1

        checks += 1
        ed = 0.0
        for i in range(idx.I):
            for j in range(idx.J):
                ed += D["edu_weight"][i, j] * _np.sum(X[i, j, t])
        if ed < D["edu_min"][t]: counts["educación"] += 1

        row = {"t":t}
        for k in types:
            row[k] = counts.get(k, 0)
        rows.append(row)
        ok = 100.0 * (1.0 - (sum(counts.values())/max(1,checks)))
        perc_ok.append({"t":t, "%_satisDates": ok})

    violM = _pd.DataFrame(rows, columns=["t"]+types)
    perc_ok = _pd.DataFrame(perc_ok)
    return violM, perc_ok

# ----------------------------------------------------------------------
# G) Orchestrator of Plan 3.2
# ----------------------------------------------------------------------
def run_extended_plan_32(top_k=6,
                         ahp_pairwise=None,
                         ahp_cols=("Minimum Cost","Penalty","Time (s, use_wrappers=True)"),
                         use_wrappers=True,
                         wrapper_kwargs=None):
    """Comprehensive execution of the recommended plan 3.2 after the baseline benchmark."""
    base = Path("results")
    base_csv = base / "benchmark_results.csv"
    if not base_csv.exists():
        print("[NOTICE] Results/benchmark_results.csv not found. Please run run_benchmark() first..")
        return

    # 1) Select Top-K by fitness of the base benchmark
    top_algs = _plus__select_top_algorithms(base_csv, k=top_k)
    print("[Plan 3.2] Selected algorithms for scenarios:", top_algs)

    # 2) Running narrative scenarios
    scenarios = _plus__scenario_defs()
    summary_per_scn = {}
    best_vectors_per_scn = {}
    for sc in scenarios:
        dfr, bestv, viol, paths = _plus__run_scenario(sc["name"], sc["ov"], sc["mult"],
                                                      alg_names=top_algs,
                                                      use_wrappers=use_wrappers,
                                                      wrapper_kwargs=wrapper_kwargs)
        summary_per_scn[sc["name"]] = dfr
        best_vectors_per_scn[sc["name"]] = bestv

    # 3) Reliability and resilience metrics (aggregated over scenarios)
    (base/"scenarios").mkdir(parents=True, exist_ok=True)
    df_rr = _plus__compute_reliability_resilience(summary_per_scn)
    df_rr.to_csv(base/"scenarios"/"summary_reliability_resilience.csv", index=False)

    # 4) AHP (default on minimization columns) in normal scenario
    dfn = summary_per_scn["Normal"].copy()
    score, w = _plus__ahp_score(dfn, list(ahp_cols), pairwise=ahp_pairwise, weights=None, invert_minimization=True)
    dfn["AHP Score"] = score
    dfn = dfn.sort_values(by="AHP Score", ascending=False)
    dfn.to_csv(base/"scenarios"/"ahp_ranking_normal.csv", index=False)
    with open(base/"scenarios"/"ahp_weights.txt","w", encoding="utf-8") as fh:
        fh.write("AHP weights used on columns {}: {}\n".format(list(ahp_cols), _np.round(w,4).tolist()))

    # 5) Pareto & HV in Normal scenario
    pts = summary_per_scn["Normal"][["Minimum Cost","Penalty"]].to_numpy()
    ref = _np.array([pts[:,0].max()*1.05, pts[:,1].max()*1.05], dtype=float)
    hv, F = _plus__hypervolume(pts, ref)
    with open(base/"scenarios"/"hypervolume_normal.txt","w", encoding="utf-8") as fh:
        fh.write(f"Hipervolumen (Normal) = {hv:.6f} with ref = {ref.tolist()}\n")
    _plt.figure(figsize=(7,6))
    _plt.scatter(pts[:,0], pts[:,1], alpha=0.5, label="Todos")
    mask = _plus__pareto_mask(pts)
    _plt.scatter(pts[mask,0], pts[mask,1], label="No dominados")
    _plt.xlabel("cost"); _plt.ylabel("penalty"); _plt.title("Escenario Normal • Pareto y HV")
    _plt.legend()
    _plus__savefig(base/"scenarios"/"pareto_hv_normal")

    # 6) Feasibility traceability (violations per period and % satisDates) for the best AHP
    best_algo = dfn.iloc[0]["Algorithm"]
    sc = [s for s in scenarios if s["name"]=="Normal"][0]
    cfg = ModelConfig()
    for k,v in sc["ov"].items(): setattr(cfg,k,v)
    idx = Indexer(cfg.n_sources, cfg.n_sectors, cfg.n_periods)
    base_eval = ScenarioModelEvaluator(cfg,
                                       demand_mult=sc["mult"]["demand"],
                                       budget_mult=sc["mult"]["budget"],
                                       avail_mult=sc["mult"]["avail"],
                                       micro_mult=sc["mult"]["micro"])
    evaluator = (FeasibleFirstAdaptiveEvaluator(base_eval, **(wrapper_kwargs or {}))
                 if use_wrappers else base_eval)
    best_vec = _np.load(base/"scenarios"/"Normal"/ f"best_vector_{best_algo}.npy")

    violM, perc_ok = _plus__violations_matrix_by_period(evaluator, cfg, idx, best_vec)
    violM.to_csv(base/"scenarios"/"violations_matrix_best_normal.csv", index=False)
    _plt.figure(figsize=(9,4.5))
    for col in violM.columns[1:]:
        _plt.plot(violM["t"], violM[col], label=col)
    _plt.xlabel("Periodo t"); _plt.ylabel("withteo de violations")
    _plt.title(f"violations por tipo • {best_algo} • Normal")
    _plt.legend(fontsize=8, ncol=3)
    _plus__savefig(base/"scenarios"/"violations_by_type_over_time_best_normal")

    _plt.figure(figsize=(7,4.2))
    _plt.plot(perc_ok["t"], perc_ok["%_satisDates"], marker="o")
    _plt.ylim(0,100); _plt.ylabel("% satisDates"); _plt.xlabel("Periodo t")
    _plt.title(f"% of satisDates restrictions • {best_algo} • Normal")
    _plus__savefig(base/"scenarios"/"%_restricciones_satisDates_best_normal")

    # 7) Reference Metadata (README)
    refs = """
    Methodological references (traceability):
    - AHP: Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
    - Pareto/Frentes y HV: Zitzler, E., & Thiele, L. (1999). Multiobjective evolutionary algorithms:
      A comparative case study. PPSN V.
    - Exploration-intensification hybrids: Talbi, E.-G. (2002). A taxonomy of hybrid metaheuristics.
      Journal of Heuristics.
    - VNS: Mladenović, N., & Hansen, P. (1997). Variable neighborhood search.
      Computers & Operations Research.
    - Tabu Search: Glover, F. (1989). Tabu Search—Part I. ORSA Journal on Computing.
    """
    (base/"scenarios"/"README_PLAN_3_2.txt").write_text(refs, encoding="utf-8")

    print("[Plan 3.2] Executed and saved in results/scenarios/.")

# ----------------------------------------------------------------------
# Self-execution of Plan 3.2 after the 25-view extension
# ----------------------------------------------------------------------
if __name__ == "__wm_base__":  # disabled in monolithic version with PLUS
    try:
        # Run the plan with default parameters and wrappers enabled
        run_extended_plan_32(top_k=6,
            ahp_pairwise=[[1,3,2],
                          [1/3,1,1/2],
                          [1/2,2,1]],
            use_wrappers=True,
            wrapper_kwargs=dict(big_M=1e12,
                                alpha_init=1.0, alpha_min=0.25, alpha_max=16.0,
                                window=200, up_ratio=0.60, down_ratio=0.20,
                                step_up=1.10, step_down=0.95)
        )
    except Exception as e:
        print("[NOTICE][Plan 3.2] It was not possible to execute the recommended plan 3.2:", e)


####====  AQUÍ VOY ====####
# ================================================================
# ===  PLUS SECTION: Verification of feasibility and "feasible-first" ===
#     (Added at the end without altering the previous codebase)
# ================================================================
# This block adds:
#   - _plus__necessary_withditions(cfg, D)
#   - _plus__solve_feasibility_lp(cfg, D)  [HiGHS]
#   - _plus__run_feasible_first(cfg, out_dir, algos)  [injects feasible seed]
#   - Main CLI if this file is invoked directly.
from typing import Dict, Any, Tuple
import json as _json
import numpy as _np
import pandas as _pd

def _plus__necessary_conditions(cfg: ModelConfig, D: Dict[str, _np.ndarray]) -> Dict[str,Any]:
    I, J, T = cfg.n_sources, cfg.n_sectors, cfg.n_periods
    env_frac = cfg.env_frac
    avail = D["avail"]             # (T,I)
    demand = D["demand"]           # (T,J)
    demand_coverage_min = cfg.demand_coverage_min
    strategic_min = D["strategic_min"]
    budget_max = D["budget_max"]
    infra_cap = D["infra_cap"]
    cost_i = D["cost"]
    trans_cost = D["trans_cost"]
    trans_coef = D["trans_coef"]
    conc_i = D["micro_withc_source"]
    micro_max_sector = D["micro_max_sector"]
    qmon_weight = D["qmon_weight"]
    qmon_min = D["qmon_min"]
    maint_weight = D["maint_weight"]
    maint_min = D["maint_min"]
    edu_weight_sector = D["edu_weight_sector"]
    edu_min = D["edu_min"]
    capacity_total = D["treat_cap"]
    capacity_pre = D["pre_cap"]
    capacity_primary = D["primary_cap"]
    capacity_secondary = D["sewithdary_cap"]
    capacity_tertiary = D["tertiary_cap"]

    report = {"periods": []}
    eps = 1e-9

    supply_eff = env_frac * _np.sum(avail, axis=1)                  # (T,)
    demand_min = demand_coverage_min * _np.sum(demand, axis=1)      # (T,)
    strategic_total = _np.sum(strategic_min, axis=1)                # (T,)
    required_flow = demand_min + strategic_total                    # (T,)

    unit_cost_min = _np.min(cost_i[:,None] + trans_cost)
    unit_coef_min = _np.min(trans_coef)

    micro_sector_ok = [bool(_np.any(conc_i <= micro_max_sector[j])) for j in range(J)]

    for t in range(T):
        caps = [capacity_total[t], capacity_pre[t], capacity_primary[t],
                capacity_secondary[t], capacity_tertiary[t]]
        period = dict(
            t=t,
            supply_eff=float(supply_eff[t]),
            required_min_flow=float(required_flow[t]),
            cond_supply_vs_required=bool(supply_eff[t] + eps >= required_flow[t]),
            caps=[float(c) for c in caps],
            cond_caps_vs_required=bool(_np.min(caps) + eps >= required_flow[t]),
            min_cost_needed=float(unit_cost_min * required_flow[t]),
            budget_max=float(budget_max[t]),
            cond_budget=bool(budget_max[t] + eps >= unit_cost_min * required_flow[t]),
            min_coef_needed=float(unit_coef_min * required_flow[t]),
            infra_cap=float(infra_cap[t]),
            cond_infra=bool(infra_cap[t] + eps >= unit_coef_min * required_flow[t]),
        )
        max_qmon = supply_eff[t] * float(_np.max(qmon_weight))
        max_maint = supply_eff[t] * float(_np.max(maint_weight))
        max_edu = supply_eff[t] * float(_np.max(edu_weight_sector))
        period.update(dict(
            max_qmon=float(max_qmon), qmon_min=float(qmon_min[t]), cond_qmon_upperbound=bool(max_qmon + eps >= qmon_min[t]),
            max_maint=float(max_maint), maint_min=float(maint_min[t]), cond_maint_upperbound=bool(max_maint + eps >= maint_min[t]),
            max_edu=float(max_edu), edu_min=float(edu_min[t]), cond_edu_upperbound=bool(max_edu + eps >= edu_min[t]),
        ))
        report["periods"].append(period)

    report["micro_sector_ok"] = micro_sector_ok
    report["all_micro_sectors_have_eligible_source"] = all(micro_sector_ok)
    return report

def _plus__solve_feasibility_lp(cfg: ModelConfig, D: Dict[str,_np.ndarray]) -> Dict[str,Any]:
    try:
        from scipy.optimize import linprog
    except Exception as e:
        return {"status": 9, "message": f"SciPy/HiGHS not available: {e}", "slack_sum": None}

    I, J, T = cfg.n_sources, cfg.n_sectors, cfg.n_periods
    env_frac = cfg.env_frac
    avail = D["avail"]
    demand = D["demand"]
    demand_coverage_min = cfg.demand_coverage_min
    strategic_min = D["strategic_min"]
    budget_max = D["budget_max"]
    infra_cap = D["infra_cap"]
    cost_i = D["cost"]
    trans_cost = D["trans_cost"]
    trans_coef = D["trans_coef"]
    conc_i = D["micro_withc_source"]
    micro_max_sector = D["micro_max_sector"]
    qmon_weight = D["qmon_weight"]
    qmon_min = D["qmon_min"]
    maint_weight = D["maint_weight"]
    maint_min = D["maint_min"]
    edu_weight_sector = D["edu_weight_sector"]
    edu_min = D["edu_min"]
    capacity_total = D["treat_cap"]
    capacity_pre = D["pre_cap"]
    capacity_primary = D["primary_cap"]
    capacity_secondary = D["sewithdary_cap"]
    capacity_tertiary = D["tertiary_cap"]
    pair_upper_bounds = cfg.pair_upper_bounds
    global_upper_bound = cfg.global_upper_bound
    drought_frac = cfg.drought_resilience_frac

    def idx(i,j,t): return t*I*J + i*J + j

    n_x = I*J*T
    sizes = dict(
        env=I*T, dem=J*T, strat=J*T,
        cap_total=T, cap_pre=T, cap_primary=T, cap_secondary=T, cap_tertiary=T,
        pair=I*J*T, glob=I*J*T, drought=T, budget=T, infra=T, micro=J*T, qmon=T, maint=T, edu=T
    )
    offset = {}; start = n_x
    for k, s in sizes.items():
        offset[k] = start; start += s
    n_vars = start
    c = _np.zeros(n_vars)
    for k, s in sizes.items():
        c[offset[k]:offset[k]+s] = 1.0

    bounds = [(0.0, None)]*n_x
    for k, s in sizes.items():
        bounds.extend([(0.0, None)]*s)

    A_ub, b_ub = [], []

    # (1) Ambiental
    for t in range(T):
        for i in range(I):
            row = _np.zeros(n_vars)
            for j in range(J):
                row[idx(i,j,t)] = 1.0
            row[offset['env'] + t*I + i] = -1.0
            A_ub.append(row); b_ub.append(env_frac * avail[t,i])

    # (2) Demand mínima
    for t in range(T):
        for j in range(J):
            row = _np.zeros(n_vars)
            for i in range(I): row[idx(i,j,t)] = -1.0
            row[offset['dem'] + t*J + j] = -1.0
            A_ub.append(row); b_ub.append(-demand_coverage_min * demand[t,j])

    # (3) Strategic reserve
    for t in range(T):
        for j in range(J):
            row = _np.zeros(n_vars)
            for i in range(I): row[idx(i,j,t)] = -1.0
            row[offset['strat'] + t*J + j] = -1.0
            A_ub.append(row); b_ub.append(-strategic_min[t,j])

    # (4) Capacities (total and sub-stages)
    def add_cap_block(name, cap_array):
        for t in range(T):
            row = _np.zeros(n_vars)
            for i in range(I):
                for j in range(J):
                    row[idx(i,j,t)] = 1.0
            row[offset[name] + t] = -1.0
            A_ub.append(row); b_ub.append(cap_array[t])
    add_cap_block('cap_total', capacity_total)
    add_cap_block('cap_pre', capacity_pre)
    add_cap_block('cap_primary', capacity_primary)
    add_cap_block('cap_secondary', capacity_secondary)
    add_cap_block('cap_tertiary', capacity_tertiary)

    # (5) Dimension per pair
    for t in range(T):
        for i in range(I):
            for j in range(J):
                row = _np.zeros(n_vars)
                row[idx(i,j,t)] = 1.0
                row[offset['pair'] + t*(I*J) + i*J + j] = -1.0
                A_ub.append(row); b_ub.append(pair_upper_bounds[i,j])

    # (6) Global quota
    for t in range(T):
        for i in range(I):
            for j in range(J):
                row = _np.zeros(n_vars)
                row[idx(i,j,t)] = 1.0
                row[offset['glob'] + t*(I*J) + i*J + j] = -1.0
                A_ub.append(row); b_ub.append(global_upper_bound)

    # (7) Total drought
    for t in range(T):
        row = _np.zeros(n_vars)
        for i in range(I):
            for j in range(J):
                row[idx(i,j,t)] = -1.0
        row[offset['drought'] + t] = -1.0
        A_ub.append(row); b_ub.append(-drought_frac * _np.sum(demand[t,:]))

    # (8) Budget
    for t in range(T):
        row = _np.zeros(n_vars)
        for i in range(I):
            for j in range(J):
                row[idx(i,j,t)] = cost_i[i] + trans_cost[i,j]
        row[offset['budget'] + t] = -1.0
        A_ub.append(row); b_ub.append(budget_max[t])

    # (9) Infrastructure
    for t in range(T):
        row = _np.zeros(n_vars)
        for i in range(I):
            for j in range(J):
                row[idx(i,j,t)] = trans_coef[i,j]
        row[offset['infra'] + t] = -1.0
        A_ub.append(row); b_ub.append(infra_cap[t])

    # (10) Micro by sector
    for t in range(T):
        for j in range(J):
            row = _np.zeros(n_vars)
            for i in range(I):
                row[idx(i,j,t)] = (conc_i[i] - micro_max_sector[j])
            row[offset['micro'] + t*J + j] = -1.0
            A_ub.append(row); b_ub.append(0.0)

    # (11) Minimum monitoring
    for t in range(T):
        row = _np.zeros(n_vars)
        for i in range(I):
            for j in range(J):
                row[idx(i,j,t)] = -qmon_weight[i,j]
        row[offset['qmon'] + t] = -1.0
        A_ub.append(row); b_ub.append(-qmon_min[t])

    # (12) Minimum maintenance
    for t in range(T):
        row = _np.zeros(n_vars)
        for i in range(I):
            for j in range(J):
                row[idx(i,j,t)] = -maint_weight[i,j]
        row[offset['maint'] + t] = -1.0
        A_ub.append(row); b_ub.append(-maint_min[t])

    # (13) Minimum education
    for t in range(T):
        row = _np.zeros(n_vars)
        for j in range(J):
            for i in range(I):
                row[idx(i,j,t)] = -edu_weight_sector[j]
        row[offset['edu'] + t] = -1.0
        A_ub.append(row); b_ub.append(-edu_min[t])

    A_ub = _np.vstack(A_ub); b_ub = _np.asarray(b_ub, float)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    out = {"status": int(res.status), "message": res.message, "slack_sum": float(res.fun) if res.success else None}
    if res.success:
        x = res.x[:n_x]
        # Reconstruct as (I,J,T). Index idx(i,j,t)=t*I*J + i*J + j already corresponds to order C (default)
        X = x.reshape((T, I, J)).transpose(1,2,0)  # (I,J,T)
        out["x"] = X
    return out

def _plus__run_feasible_first(cfg: ModelConfig, out_dir: str, algos=("SA","SCA","GA")):
    import pathlib
    paths = ensure_dirs(pathlib.Path(out_dir))
    fix_all_seeds(cfg.base_seed)

    evaluator = ModelEvaluator(cfg)
    D = evaluator.data

    # (i) Necessary conditions
    report = _plus__necessary_conditions(cfg, D)
    with open(str(pathlib.Path(out_dir) / "necessary_withditions.json"), "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)

    # (ii) Feasibility LP
    res = _plus__solve_feasibility_lp(cfg, D)
    with open(str(pathlib.Path(out_dir) / "feasibility_lp_result.json"), "w", encoding="utf-8") as f:
        _json.dump({k: res.get(k) for k in ["status","message","slack_sum"]}, f, ensure_ascii=False, indent=2)

    idx = Indexer(cfg.n_sources, cfg.n_sectors, cfg.n_periods)

    if res.get("x") is not None and abs(res.get("slack_sum", 1e9)) <= 1e-8:
        X0 = res["x"]
        x0 = idx.to_vector(X0)
        np.save(str(pathlib.Path(out_dir) / "x_feasible_seed.npy"), x0)
    else:
        if res.get("x") is not None:
            x0 = idx.to_vector(res["x"])
        else:
            rng = np.random.default_rng(cfg.base_seed+123)
            x0 = random_solution(cfg, idx, rng)
        np.save(str(pathlib.Path(out_dir) / "x_seed.npy"), x0)

    alg_map = {
        "GA": GA_optimize, "PSO": PSO_optimize, "DE": DE_optimize, "SA": SA_optimize,
        "HS": HS_optimize, "ABC": ABC_optimize, "VNS": VNS_optimize, "TS": TS_optimize,
        "ES": ES_optimize, "EP": EP_optimize, "FA": FA_optimize, "CS": CS_optimize,
        "WOA": WOA_optimize, "BA": BA_optimize, "TLBO": TLBO_optimize, "MA": MA_optimize,
        "GSA": GSA_optimize, "ICA": ICA_optimize, "BBO": BBO_optimize, "ALO": ALO_optimize,
        "QPSO": QPSO_optimize, "DA": DA_optimize, "ACOR": ACOR_optimize, "GWO": GWO_optimize,
        "SCA": SCA_optimize, "HHO": HHO_optimize
    }

    results = []
    histories = {}

    for name in algos:
        if name not in alg_map: 
            continue
        rngs = make_rngs(cfg.base_seed, [name])
        rng = rngs[name]

        # Inject viable seed into the first individual
        orig_random_solution = random_solution
        inject = {"done": False}
        def _inject_rand(cfg_, idx_, rng_):
            if not inject["done"]:
                inject["done"] = True
                return x0.copy()
            return orig_random_solution(cfg_, idx_, rng_)
        globals()['random_solution'] = _inject_rand

        try:
            best_x, best_f, hist = alg_map[name](evaluator, cfg, idx, rng)
        finally:
            globals()['random_solution'] = orig_random_solution

        ev = evaluator.evaluate(best_x, idx)
        results.append(dict(Algorithm=name, BestFitness=best_f, MinimumCost=ev.cost,
                            Penalty=ev.penalty, Violations=ev.n_violations))
        histories[name] = hist
        np.save(str(pathlib.Path(out_dir) / f"best_vector_{name}.npy"), best_x)

    df = _pd.DataFrame(results).sort_values("Penalty")
    df.to_csv(str(pathlib.Path(out_dir) / "benchmark_results_feasible_first.csv"), index=False)

    # Stories
    hist_recs = []
    for k, seq in histories.items():
        for it, val in enumerate(seq):
            hist_recs.append(dict(Algorithm=k, Iteration=it, BestFitness=val))
    _pd.DataFrame(hist_recs).to_csv(str(pathlib.Path(out_dir) / "histories_feasible_first.csv"), index=False)

    return df

# ------------------------------------------------------------
# Main CLI (optional)
# ------------------------------------------------------------
if __name__ == "__main__":  # Disabled in monolithic version with PLUS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_feasible", help="Output folder")
    ap.add_argument("--algos", default="SA,SCA,GA", help="Algorithms separated by commas")
    args = ap.parse_args()

    cfg = ModelConfig()
    import pathlib; pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    df = _plus__run_feasible_first(cfg, args.out, algos=[s.strip() for s in args.algos.split(",") if s.strip()])
    print(df.to_string(index=False))
# ============================================================
# === ROBUST ROUTE OVERRIDES AND FIGURE SAVINGS    ===
# ============================================================
def ensure_dirs(base="results"):
    """
    Creates and returns a dictionary of ABSOLUTE paths: base, figs, logs.
    Robust override to ensure disk writing.
    """
    from pathlib import Path as _P
    base = _P(base).expanduser().resolve()
    figs = base / "figs"
    logs = base / "logs"
    figs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    paths = {"base": base, "figs": figs, "logs": logs}
    print(f"[ensure_dirs] Wearing base={base}")
    return paths

def savefig_all(base_or_fig, maybe_base=None):
    """
    save the current figure or the given figure in PNG/PDF/SVG with the same basename.
    Valid uses:
      - savefig_all(basepath)             # uses plt.gcf()
      - savefig_all(fig, basepath)
    Close the figure after save.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path as _P
    fig = None
    base = None
    try:
        if maybe_base is None:
            base = _P(base_or_fig)
            fig = plt.gcf()
        else:
            fig = base_or_fig
            base = _P(maybe_base)
        base = base.expanduser().resolve()
        base.parent.mkdir(parents=True, exist_ok=True)
        for suf in (".png",".pdf",".svg"):
            fpath = base.with_suffix(suf)
            try:
                fig.savefig(str(fpath), dpi=200, bbox_inches="tight")
                print(f"[savefig_all] savedo: {fpath}")
            except Exception as e:
                print(f"[savefig_all][ADVERTENCIA] No se pudo save {fpath}: {e}")
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            pass

# ============================================================
# === Executable: benchmark + 25 views            ===
# ============================================================
if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print("[ADVERTENCIA] run_benchmark falló:", e)
    try:
        generate_extra_visualizations_25()
    except Exception as e:
        print("[ADVERTENCIA] Visualizaciones extra no generadas:", e)