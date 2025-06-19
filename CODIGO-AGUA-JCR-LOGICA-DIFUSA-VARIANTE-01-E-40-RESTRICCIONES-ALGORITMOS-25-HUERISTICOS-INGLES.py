# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:03:11 2025

@author: Jaime Aguilar Ortiz
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete code that integrates fuzzy logic for water management (with its functions, parameters, constants, and restrictions)
and the application of the 25 methods listed in "TIPOS-ALGORITMOS-GENETICOS-HEURISTICOS-META.docx". It includes:
 1. Genetic Algorithm (GA)
 2. Evolution Strategy (ES)
 3. Evolutionary Programming (EP)
 4. Simulated Annealing (SA)
 5. Tabu Search (TS)
 6. Ant Colony Optimization (ACO)
 7. PSO (Particle Swarm Optimization)
 8. DE (Differential Evolution)
 9. Artificial Bee Colony (ABC)
10. Variable Neighborhood Search (VNS)
11. Memetic Algorithm (MA)
12. Scatter Search (SS)
13. Harmony Search (HS)
14. Firefly Algorithm (FA)
15. Cuckoo Search (CS)
16. Gravitational Search Algorithm (GSA)
17. Whale Optimization Algorithm (WOA)
18. Bat Algorithm (BA)
19. Imperialist Competitive Algorithm (ICA)
20. TLBO (Teaching-Learning-Based Optimization)
21. Cultural Algorithm (CA)
22. Biogeography-Based Optimization (BBO)
23. Ant Lion Optimizer (ALO)
24. QPSO (Quantum-behaved PSO)
25. Dragonfly Algorithm (DA)

At the end, a comparative table is generated with the following indicators:
- Algorithm  
- Optimal Value  
- Minimum Cost  
- Best Fitness  
- Execution Time (s)

Only the graphs related to fuzzy logic are also displayed.
"""

####################################
# FUZZY LOGIC FUNCTIONS
####################################
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from math import pi, exp, log, cos

def get_membership_functions(x, min_val, max_val, type="triangular"):
    """
    Returns a dictionary containing 'low', 'medium', and 'high' membership functions
    based on the specified type.
    """
    mid = (min_val + max_val) / 2.0
    if type == "triangular":
        low = fuzz.trimf(x, [min_val, min_val, mid])
        medium = fuzz.trimf(x, [min_val, mid, max_val])
        high = fuzz.trimf(x, [mid, max_val, max_val])
    elif type == "trapezoidal":
        low = fuzz.trapmf(x, [min_val, min_val, min_val + (max_val - min_val)/4, min_val + (max_val - min_val)/3])
        medium = fuzz.trapmf(x, [min_val + (max_val - min_val)/6, mid, mid, max_val - (max_val - min_val)/6])
        high = fuzz.trapmf(x, [max_val - (max_val - min_val)/3, max_val - (max_val - min_val)/4, max_val, max_val])
    elif type == "gaussian":
        sigma = (max_val - min_val) / 6.0
        low = fuzz.gaussmf(x, min_val + (max_val - min_val)/4, sigma)
        medium = fuzz.gaussmf(x, mid, sigma)
        high = fuzz.gaussmf(x, max_val - (max_val - min_val)/4, sigma)
    else:
        raise ValueError("Unrecognized membership function type.")
    return {"low": low, "medium": medium, "high": high}

def defuzzify_variable(x, functions, nominal_value, method='centroid'):
    """
    Defuzzifies a variable using the provided membership functions and defuzzification method.
    """
    low_degree = fuzz.interp_membership(x, functions["low"], nominal_value)
    medium_degree = fuzz.interp_membership(x, functions["medium"], nominal_value)
    high_degree = fuzz.interp_membership(x, functions["high"], nominal_value)
    low_clipped = np.fmin(functions["low"], low_degree)
    medium_clipped = np.fmin(functions["medium"], medium_degree)
    high_clipped = np.fmin(functions["high"], high_degree)
    aggregated = np.fmax(low_clipped, np.fmax(medium_clipped, high_clipped))
    crisp_value = fuzz.defuzz(x, aggregated, method)
    return crisp_value

def defuzzify_cost(cost_params, type="triangular", method='centroid', num_points=1000):
    """
    Defuzzifies cost using the specified fuzzy cost function parameters.
    """
    x_cost = np.linspace(cost_params[0], cost_params[2], num_points)
    if type == "triangular":
        mf = fuzz.trimf(x_cost, [cost_params[0], cost_params[1], cost_params[2]])
    elif type == "trapezoidal":
        mf = fuzz.trapmf(x_cost, [cost_params[0], cost_params[0], cost_params[1], cost_params[2]])
    elif type == "gaussian":
        sigma = (cost_params[2] - cost_params[0]) / 6.0
        mf = fuzz.gaussmf(x_cost, cost_params[1], sigma)
    else:
        raise ValueError("Unrecognized cost function type.")
    crisp_cost = fuzz.defuzz(x_cost, mf, method)
    return crisp_cost

####################################
# PARAMETERS AND CONFIGURATION
####################################
type_function = "triangular"
defuzz_method = "centroid"

# Ranges for water availability per source
avail_ranges = {
    1: {"min": 0, "max": 20000, "nominal": 12000},
    2: {"min": 0, "max": 15000, "nominal": 8000},
    3: {"min": 0, "max": 10000, "nominal": 5000}
}

# Ranges for water demand per sector
demand_ranges = {
    1: {"min": 0, "max": 10000, "nominal": 6000},
    2: {"min": 0, "max": 20000, "nominal": 12000},
    3: {"min": 0, "max": 8000, "nominal": 4000}
}

# Fuzzy operational costs per source (minimum, nominal, maximum)
cost_params = {
    1: (0.025, 0.03, 0.035),
    2: (0.015, 0.02, 0.025),
    3: (0.01, 0.015, 0.02)
}

# Fuzzy treatment capacity (in cubic meters)
treatment_capacity_fuzzy = (55000, 60000, 65000)

# Temporal dimension and factors
n_periods = 2
availability_factors = {1: 1.0, 2: 0.9}
demand_factors = {1: 1.0, 2: 1.1}

# Defuzzification calculations for availabilities
availabilities = {}
for i, params in avail_ranges.items():
    x_avail = np.arange(params["min"], params["max"] + 1, 1)
    funcs_avail = get_membership_functions(x_avail, params["min"], params["max"], type_function)
    crisp_value = defuzzify_variable(x_avail, funcs_avail, params["nominal"], defuzz_method)
    availabilities[i] = {}
    for t in range(1, n_periods + 1):
        availabilities[i][t] = availability_factors[t] * crisp_value

# Defuzzification calculations for demands
demands = {}
for j, params in demand_ranges.items():
    x_demand = np.arange(params["min"], params["max"] + 1, 1)
    funcs_demand = get_membership_functions(x_demand, params["min"], params["max"], type_function)
    crisp_value = defuzzify_variable(x_demand, funcs_demand, params["nominal"], defuzz_method)
    demands[j] = {}
    for t in range(1, n_periods + 1):
        demands[j][t] = demand_factors[t] * crisp_value

# Defuzzification of operational costs for each source
costs = {}
for i, params in cost_params.items():
    costs[i] = defuzzify_cost(params, type_function, defuzz_method)

x_cap = np.linspace(treatment_capacity_fuzzy[0], treatment_capacity_fuzzy[2], 1000)
treatment_capacity = fuzz.defuzz(
    x_cap,
    fuzz.trimf(x_cap, [treatment_capacity_fuzzy[0],
                       treatment_capacity_fuzzy[1],
                       treatment_capacity_fuzzy[2]]),
    defuzz_method
)

# Environmental restriction: extract up to a given percentage of availability
environmental_percentage = 0.9

# Additional parameters
A = {1: 10000, 2: 10000, 3: 10000}      # Availability per source
D = {1: 5000, 2: 5000, 3: 5000}          # Minimum demand per sector
Ct = 60000                             # Total treatment capacity
B = 1000000                            # Maximum budget
Cij = {1: 0.03, 2: 0.02, 3: 0.015}       # Water cost per source
COij = {1: 0.03, 2: 0.02, 3: 0.01}
CEij = {1: 0.01, 2: 0.008, 3: 0.005}
CMij = {1: 0.02, 2: 0.015, 3: 0.01}
CENij = {1: 0.02, 2: 0.015, 3: 0.01}
Aij = {(1,1): 1.0, (1,2): 1.3, (1,3): 1.2,
       (2,1): 0.2, (2,2): 0.3, (2,3): 0.2,
       (3,1): 0.4, (3,2): 0.5, (3,3): 0.4}
Cinfra = 80000
Tr = {(1,1): 1.0, (1,2): 1.1, (1,3): 1.0,
      (2,1): 1.3, (2,2): 1.3, (2,3): 1.2,
      (3,1): 1.5, (3,2): 1.4, (3,3): 1.3}
R_drought = 0.20
R_strategic = 10000
L_norm = 10000
B_hidro = 1
Ds = {1: 5000, 2: 5000, 3: 5000}  # For equity during scarcity

Amax = 20000  # Maximum limit for aquifer extraction (in cubic meters)

# Restrictions 44 to 50
M_max = {1: 5.0, 2: 10.0, 3: 8.0}
M = {(1,1): 3.5, (1,2): 7.0, (1,3): 6.5,
     (2,1): 4.0, (2,2): 8.5, (2,3): 7.0,
     (3,1): 3.8, (3,2): 9.0, (3,3): 7.5}

F = {(1,1): 1.5, (1,2): 2.5, (1,3): 2.0,
     (2,1): 1.0, (2,2): 1.8, (2,3): 1.5,
     (3,1): 0.8, (3,2): 1.2, (3,3): 1.0}
F_max = 1000000

INF = {(1,1): 0.05, (1,2): 0.08, (1,3): 0.06,
       (2,1): 0.07, (2,2): 0.09, (2,3): 0.05,
       (3,1): 0.04, (3,2): 0.06, (3,3): 0.03}
INF_max = 4000

pH_Control = [[7.2, 7.5, 7.8],
              [7.1, 7.3, 7.6],
              [6.8, 7.0, 7.2]]
pH_max = 8.5  # Assumed to be satisfied

Q_mon_min = 750
Q_mon = {(1,1): 0.18, (1,2): 0.20, (1,3): 0.22,
         (2,1): 0.16, (2,2): 0.19, (2,3): 0.21,
         (3,1): 0.17, (3,2): 0.20, (3,3): 0.23}

Minfij = [[50, 60, 55],
          [65, 70, 60],
          [55, 50, 45]]
Minf_min = 200

Emin = 2000
E_edu = [[0.0050, 0.0070, 0.0065],
         [0.0060, 0.0072, 0.0068],
         [0.0055, 0.0071, 0.0067]]

####################################
# MAPPING OF OPTIMIZATION VARIABLES
####################################
indices = [(i, j, t) for i in sorted(avail_ranges.keys())
                   for j in sorted(demand_ranges.keys())
                   for t in range(1, n_periods + 1)]
n_variables = len(indices)

def vector_to_dict(sol_vector):
    sol_dict = {}
    for idx, key in enumerate(indices):
        sol_dict[key] = sol_vector[idx]
    return sol_dict

def evaluate_solution(sol_vector):
    """
    Evaluates a solution by calculating the objective function (total cost)
    and adding penalty terms for any constraint violations.
    """
    x_sol = vector_to_dict(sol_vector)
    objective = 0.0
    for (i, j, t) in indices:
        objective += costs[i] * x_sol[(i, j, t)]
    penalty = 0.0
    pen_factor = 1e6
    for t in range(1, n_periods + 1):
        # Restriction 1: Availability per source (adjusted by environmental percentage)
        for i in avail_ranges.keys():
            source_sum = sum(x_sol[(i, j, t)] for j in demand_ranges.keys())
            source_limit = environmental_percentage * availabilities[i][t]
            if source_sum > source_limit:
                penalty += pen_factor * (source_sum - source_limit)**2
        # Restriction 2: Minimum demand per sector
        for j in demand_ranges.keys():
            sector_sum = sum(x_sol[(i, j, t)] for i in avail_ranges.keys())
            if sector_sum < demands[j][t]:
                penalty += pen_factor * (demands[j][t] - sector_sum)**2
        # Restriction 3: Treatment capacity
        total_sum = sum(x_sol[(i, j, t)] for i in avail_ranges.keys() for j in demand_ranges.keys())
        if total_sum > treatment_capacity:
            penalty += pen_factor * (total_sum - treatment_capacity)**2
        # Additional Restriction: Availability A
        for i in avail_ranges.keys():
            source_A_sum = sum(x_sol[(i, j, t)] for j in demand_ranges.keys())
            if source_A_sum > A[i]:
                penalty += pen_factor * (source_A_sum - A[i])**2
        # Additional Restriction: Minimum demand D
        for j in demand_ranges.keys():
            sector_D_sum = sum(x_sol[(i, j, t)] for i in avail_ranges.keys())
            if sector_D_sum < D[j]:
                penalty += pen_factor * (D[j] - sector_D_sum)**2
        # Additional Restriction: Treatment capacity Ct
        if total_sum > Ct:
            penalty += pen_factor * (total_sum - Ct)**2
        # Additional Restriction: Budget B
        cost_sum = 0.0
        for i in avail_ranges.keys():
            for j in demand_ranges.keys():
                cost_sum += (Cij[i] + COij[i] + CEij[i] + CMij[i] + CENij[i]) * x_sol[(i, j, t)]
        if cost_sum > B:
            penalty += pen_factor * (cost_sum - B)**2
        # Additional Restriction: Aquifer protection
        aquifer_sum = 0.0
        for i in avail_ranges.keys():
            for j in demand_ranges.keys():
                aquifer_sum += Aij[(i, j)] * x_sol[(i, j, t)]
        if aquifer_sum > Amax:
            penalty += pen_factor * (aquifer_sum - Amax)**2
        # Additional Restriction: Infrastructure capacity
        infra_sum = 0.0
        for i in avail_ranges.keys():
            for j in demand_ranges.keys():
                infra_sum += Tr[(i, j)] * x_sol[(i, j, t)]
        if total_sum > 0 and (infra_sum > Cinfra * total_sum):
            penalty += pen_factor * (infra_sum - Cinfra * total_sum)**2
        # Additional Restriction: Drought resilience
        if total_sum < R_drought:
            penalty += pen_factor * (R_drought - total_sum)**2
        # Additional Restriction: Minimum hydrological balance
        if total_sum < B_hidro:
            penalty += pen_factor * (B_hidro - total_sum)**2
        # Additional Restriction: Strategic reserves
        for j in demand_ranges.keys():
            reserve_sum = sum(x_sol[(i, j, t)] for i in avail_ranges.keys())
            if reserve_sum > R_strategic:
                penalty += pen_factor * (reserve_sum - R_strategic)**2
    # Global Restriction: Norm limit
    for key in indices:
        if x_sol[key] > L_norm:
            penalty += pen_factor * (x_sol[key] - L_norm)**2
    # New Restrictions (44 to 50)
    for t in range(1, n_periods + 1):
        # Restriction 44: Micropollutants (evaluating the average concentration per sector)
        for j in demand_ranges.keys():
            sector_sum = sum(x_sol[(i, j, t)] for i in avail_ranges.keys())
            if sector_sum > 0:
                avg_concentration = sum(M[(i, j)] * x_sol[(i, j, t)] for i in avail_ranges.keys()) / sector_sum
                if avg_concentration > M_max[j]:
                    penalty += pen_factor * (avg_concentration - M_max[j])**2
        # Restriction 45: Efficient use of financial resources
        financial_sum = 0.0
        for i in avail_ranges.keys():
            for j in demand_ranges.keys():
                financial_sum += F[(i, j)] * x_sol[(i, j, t)]
        if financial_sum > F_max:
            penalty += pen_factor * (financial_sum - F_max)**2
        # Restriction 46: Prevention of infiltrations
        infiltration_sum = 0.0
        for i in avail_ranges.keys():
            for j in demand_ranges.keys():
                infiltration_sum += INF[(i, j)] * x_sol[(i, j, t)]
        if infiltration_sum > INF_max:
            penalty += pen_factor * (infiltration_sum - INF_max)**2
        # Restriction 48: Quality monitoring
        monitor_sum = 0.0
        for i in avail_ranges.keys():
            for j in demand_ranges.keys():
                monitor_sum += Q_mon[(i, j)] * x_sol[(i, j, t)]
        if monitor_sum < Q_mon_min:
            penalty += pen_factor * (Q_mon_min - monitor_sum)**2
        # Restriction 49: Infrastructure maintenance
        maintenance_sum = 0.0
        for (i, j, _) in indices:
            maintenance_sum += Minfij[i-1][j-1] * x_sol[(i, j, t)]
        if maintenance_sum < Minf_min:
            penalty += pen_factor * (Minf_min - maintenance_sum)**2
        # Restriction 50: Education and awareness
        education_sum = 0.0
        for (i, j, _) in indices:
            education_sum += E_edu[i-1][j-1] * x_sol[(i, j, t)]
        if education_sum < Emin:
            penalty += pen_factor * (Emin - education_sum)**2
    return objective + penalty

#####################################
# FUNCTION TO CALCULATE MINIMUM COST
#####################################
def calculate_cost(sol_vector):
    x_sol = vector_to_dict(sol_vector)
    return sum(costs[i] * x_sol[(i, j, t)] for (i, j, t) in indices)

####################################
# PARAMETERS FOR THE ALGORITHMS
####################################
population_size = 100
generations = 200
lower_bound = 0
upper_bound = L_norm

##############################################
# IMPLEMENTATION OF THE ALGORITHMS
##############################################

# 1. Genetic Algorithm (GA)
def genetic_algorithm():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    best_sol = None
    best_fitness = float('inf')
    for gen in range(generations):
        fitnesses = [evaluate_solution(ind) for ind in population]
        for i, ind in enumerate(population):
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_sol = ind.copy()
        new_population = [best_sol.copy()]  # Elitism
        while len(new_population) < population_size:
            i1, i2 = random.sample(range(len(population)), 2)
            parent1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
            i3, i4 = random.sample(range(len(population)), 2)
            parent2 = population[i3] if fitnesses[i3] < fitnesses[i4] else population[i4]
            if random.random() < 0.8:
                child1, child2 = parent1.copy(), parent2.copy()
                for i in range(n_variables):
                    if random.random() < 0.5:
                        child1[i], child2[i] = child2[i], child1[i]
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            for child in [child1, child2]:
                for i in range(n_variables):
                    if random.random() < 0.1:
                        child[i] += np.random.normal(0, L_norm * 0.1)
                        child[i] = np.clip(child[i], lower_bound, upper_bound)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
            population = new_population
    return best_sol, best_fitness

# 2. Evolution Strategy (ES)
def evolution_strategy():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    best_sol = None
    best_fitness = float('inf')
    mutation_prob = 0.3
    for gen in range(generations):
        new_population = []
        for ind in population:
            child = ind.copy()
            for i in range(n_variables):
                if random.random() < mutation_prob:
                    child[i] += np.random.normal(0, L_norm * 0.2)
                    child[i] = np.clip(child[i], lower_bound, upper_bound)
            new_population.append(child)
        population = new_population
        for ind in population:
            f = evaluate_solution(ind)
            if f < best_fitness:
                best_fitness = f
                best_sol = ind.copy()
    return best_sol, best_fitness

# 3. Evolutionary Programming (EP)
def evolutionary_programming():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    sigma = np.full((population_size, n_variables), L_norm * 0.1)
    best_sol = None
    best_fitness = float('inf')
    for gen in range(generations):
        new_population = []
        for idx, ind in enumerate(population):
            child = ind.copy()
            sigma[idx] *= np.exp(0.1 * np.random.normal(0, 1, n_variables))
            child += np.random.normal(0, sigma[idx])
            child = np.clip(child, lower_bound, upper_bound)
            new_population.append(child)
        population = new_population
        for ind in population:
            f = evaluate_solution(ind)
            if f < best_fitness:
                best_fitness = f
                best_sol = ind.copy()
    return best_sol, best_fitness

# 4. Simulated Annealing (SA)
def simulated_annealing():
    current = np.random.uniform(lower_bound, upper_bound, n_variables)
    current_fitness = evaluate_solution(current)
    best_sol = current.copy()
    best_fitness = current_fitness
    T = 1000.0
    alpha = 0.99
    iterations = generations * 10
    for it in range(iterations):
        neighbor = current + np.random.normal(0, L_norm*0.1, n_variables)
        neighbor = np.clip(neighbor, lower_bound, upper_bound)
        f_neighbor = evaluate_solution(neighbor)
        delta = f_neighbor - current_fitness
        if delta < 0 or random.random() < exp(-delta/T):
            current = neighbor.copy()
            current_fitness = f_neighbor
            if current_fitness < best_fitness:
                best_sol = current.copy()
                best_fitness = current_fitness
        T *= alpha
    return best_sol, best_fitness

# 5. Tabu Search (TS)
def tabu_search():
    current = np.random.uniform(lower_bound, upper_bound, n_variables)
    best_sol = current.copy()
    best_fitness = evaluate_solution(current)
    tabu_list = []
    tabu_size = 10
    iterations = generations * 5
    for it in range(iterations):
        neighborhood = []
        for _ in range(20):
            candidate = current + np.random.normal(0, L_norm*0.05, n_variables)
            candidate = np.clip(candidate, lower_bound, upper_bound)
            neighborhood.append(candidate)
        neighborhood = [c for c in neighborhood if not any(np.allclose(c, t, atol=1e-3) for t in tabu_list)]
        if not neighborhood:
            continue
        candidates_fitness = [evaluate_solution(c) for c in neighborhood]
        idx = np.argmin(candidates_fitness)
        current = neighborhood[idx].copy()
        current_fitness = candidates_fitness[idx]
        tabu_list.append(current.copy())
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        if current_fitness < best_fitness:
            best_sol = current.copy()
            best_fitness = current_fitness
    return best_sol, best_fitness

# 6. Ant Colony Optimization (ACO)
def ant_colony_optimization():
    archive_size = population_size
    archive = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(archive_size)]
    fitness_archive = [evaluate_solution(sol) for sol in archive]
    iterations = generations
    for it in range(iterations):
        weights = np.array([1/(f+1e-9) for f in fitness_archive])
        weights /= np.sum(weights)
        new_solutions = []
        for _ in range(archive_size):
            idx = np.random.choice(range(archive_size), p=weights)
            base = archive[idx]
            std = np.std(np.array(archive), axis=0) + 1e-9
            candidate = base + np.random.normal(0, std)
            candidate = np.clip(candidate, lower_bound, upper_bound)
            new_solutions.append(candidate)
        combined = archive + new_solutions
        combined_fitness = [evaluate_solution(sol) for sol in combined]
        idx_sorted = np.argsort(combined_fitness)
        archive = [combined[i] for i in idx_sorted[:archive_size]]
        fitness_archive = [combined_fitness[i] for i in idx_sorted[:archive_size]]
    best_idx = np.argmin(fitness_archive)
    return archive[best_idx], fitness_archive[best_idx]

# 7. Particle Swarm Optimization (PSO)
def particle_swarm_optimization():
    num_particles = population_size
    iterations = generations
    w = 0.7; c1 = 1.5; c2 = 1.5
    particles = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_particles)]
    velocities = [np.random.uniform(-abs(upper_bound - lower_bound), abs(upper_bound - lower_bound), n_variables) for _ in range(num_particles)]
    pbest = particles.copy()
    pbest_fitness = [evaluate_solution(p) for p in particles]
    gbest = pbest[np.argmin(pbest_fitness)].copy()
    for it in range(iterations):
        for i in range(num_particles):
            r1 = np.random.uniform(0,1, n_variables)
            r2 = np.random.uniform(0,1, n_variables)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], lower_bound, upper_bound)
            fitness = evaluate_solution(particles[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness
        current_best = min(pbest_fitness)
        if current_best < evaluate_solution(gbest):
            gbest = pbest[np.argmin(pbest_fitness)].copy()
    return gbest, evaluate_solution(gbest)

# 8. Differential Evolution (DE)
def differential_evolution():
    F_factor = 0.8
    CR = 0.9
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    iterations = generations
    for it in range(iterations):
        new_population = []
        for i in range(population_size):
            idxs = list(range(population_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)
            mutant = population[a] + F_factor * (population[b] - population[c])
            mutant = np.clip(mutant, lower_bound, upper_bound)
            trial = np.array([mutant[j] if random.random() < CR else population[i][j] for j in range(n_variables)])
            if evaluate_solution(trial) < evaluate_solution(population[i]):
                new_population.append(trial)
            else:
                new_population.append(population[i])
        population = new_population
    best = min(population, key=lambda ind: evaluate_solution(ind))
    return best, evaluate_solution(best)

# 9. Artificial Bee Colony (ABC)
def artificial_bee_colony():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    fitnesses = [evaluate_solution(p) for p in population]
    limit = 20
    trial = [0]*population_size
    iterations = generations
    best_sol = population[np.argmin(fitnesses)].copy()
    best_fitness = min(fitnesses)
    for it in range(iterations):
        # Employed bees phase
        for i in range(population_size):
            k = random.choice([j for j in range(population_size) if j != i])
            candidate = population[i] + np.random.uniform(-1,1, n_variables) * (population[i] - population[k])
            candidate = np.clip(candidate, lower_bound, upper_bound)
            f_candidate = evaluate_solution(candidate)
            if f_candidate < fitnesses[i]:
                population[i] = candidate.copy()
                fitnesses[i] = f_candidate
                trial[i] = 0
            else:
                trial[i] += 1
        # Onlooker bees phase
        for i in range(population_size):
            if random.random() < (1/(fitnesses[i]+1e-9)):
                k = random.choice([j for j in range(population_size) if j != i])
                candidate = population[i] + np.random.uniform(-1,1, n_variables) * (population[i] - population[k])
                candidate = np.clip(candidate, lower_bound, upper_bound)
                f_candidate = evaluate_solution(candidate)
                if f_candidate < fitnesses[i]:
                    population[i] = candidate.copy()
                    fitnesses[i] = f_candidate
                    trial[i] = 0
                else:
                    trial[i] += 1
        # Scout bees phase
        for i in range(population_size):
            if trial[i] > limit:
                population[i] = np.random.uniform(lower_bound, upper_bound, n_variables)
                fitnesses[i] = evaluate_solution(population[i])
                trial[i] = 0
        current_best = min(fitnesses)
        if current_best < best_fitness:
            best_fitness = current_best
            best_sol = population[np.argmin(fitnesses)].copy()
    return best_sol, best_fitness

# 10. Variable Neighborhood Search (VNS)
def variable_neighborhood_search():
    current = np.random.uniform(lower_bound, upper_bound, n_variables)
    best_sol = current.copy()
    best_fitness = evaluate_solution(current)
    neighborhoods = [L_norm*0.01, L_norm*0.05, L_norm*0.1]
    iterations = generations * 5
    for it in range(iterations):
        improved = False
        for delta in neighborhoods:
            candidate = current + np.random.uniform(-delta, delta, n_variables)
            candidate = np.clip(candidate, lower_bound, upper_bound)
            f_candidate = evaluate_solution(candidate)
            if f_candidate < best_fitness:
                best_sol = candidate.copy()
                best_fitness = f_candidate
                current = candidate.copy()
                improved = True
                break
        if not improved:
            current = np.random.uniform(lower_bound, upper_bound, n_variables)
    return best_sol, best_fitness

# 11. Memetic Algorithm (MA)
def memetic_algorithm():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    best_sol = None
    best_fitness = float('inf')
    local_iterations = 20
    for gen in range(generations):
        fitnesses = [evaluate_solution(ind) for ind in population]
        best_index = np.argmin(fitnesses)
        best_individual = population[best_index].copy()
        for _ in range(local_iterations):
            neighbor = best_individual.copy()
            perturbation = np.random.normal(0, L_norm * 0.05, n_variables)
            neighbor = neighbor + perturbation
            neighbor = np.clip(neighbor, lower_bound, upper_bound)
            if evaluate_solution(neighbor) < evaluate_solution(best_individual):
                best_individual = neighbor.copy()
        if evaluate_solution(best_individual) < best_fitness:
            best_fitness = evaluate_solution(best_individual)
            best_sol = best_individual.copy()
        new_population = [best_sol.copy()]
        while len(new_population) < population_size:
            i1, i2 = random.sample(range(len(population)), 2)
            parent1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]
            i3, i4 = random.sample(range(len(population)), 2)
            parent2 = population[i3] if fitnesses[i3] < fitnesses[i4] else population[i4]
            if random.random() < 0.8:
                child1, child2 = parent1.copy(), parent2.copy()
                for i in range(n_variables):
                    if random.random() < 0.5:
                        child1[i], child2[i] = child2[i], child1[i]
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            for child in [child1, child2]:
                for i in range(n_variables):
                    if random.random() < 0.1:
                        child[i] += np.random.normal(0, L_norm * 0.1)
                        child[i] = np.clip(child[i], lower_bound, upper_bound)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
            population = new_population
    return best_sol, best_fitness

# 12. Scatter Search (SS)
def scatter_search():
    archive = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    iterations = generations
    for it in range(iterations):
        new_candidates = []
        for i in range(len(archive)):
            for j in range(i+1, len(archive)):
                candidate = (archive[i] + archive[j]) / 2.0
                candidate += np.random.normal(0, L_norm*0.01, n_variables)
                candidate = np.clip(candidate, lower_bound, upper_bound)
                new_candidates.append(candidate)
        all_candidates = archive + new_candidates
        all_fitness = [evaluate_solution(sol) for sol in all_candidates]
        idx_sorted = np.argsort(all_fitness)
        archive = [all_candidates[i] for i in idx_sorted[:population_size]]
    best = archive[0]
    return best, evaluate_solution(best)

# 13. Harmony Search (HS)
def harmony_search():
    harmony_memory = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    harmony_fitness = [evaluate_solution(sol) for sol in harmony_memory]
    HMCR = 0.9; PAR = 0.3
    iterations = generations * 10
    for it in range(iterations):
        new_harmony = np.zeros(n_variables)
        for i in range(n_variables):
            if random.random() < HMCR:
                new_harmony[i] = random.choice(harmony_memory)[i]
                if random.random() < PAR:
                    new_harmony[i] += np.random.normal(0, L_norm*0.01)
            else:
                new_harmony[i] = np.random.uniform(lower_bound, upper_bound)
        new_harmony = np.clip(new_harmony, lower_bound, upper_bound)
        f_new = evaluate_solution(new_harmony)
        worst_idx = np.argmax(harmony_fitness)
        if f_new < harmony_fitness[worst_idx]:
            harmony_memory[worst_idx] = new_harmony.copy()
            harmony_fitness[worst_idx] = f_new
    best = harmony_memory[np.argmin(harmony_fitness)]
    return best, min(harmony_fitness)

# 14. Firefly Algorithm (FA)
def firefly_algorithm():
    num_fireflies = population_size
    fireflies = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_fireflies)]
    fitnesses = [evaluate_solution(f) for f in fireflies]
    alpha = 0.2; beta0 = 1.0; gamma = 1e-4
    iterations = generations
    for it in range(iterations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if fitnesses[j] < fitnesses[i]:
                    r = np.linalg.norm(fireflies[i]-fireflies[j])
                    beta = beta0 * exp(-gamma * r**2)
                    fireflies[i] = fireflies[i] + beta*(fireflies[j]-fireflies[i]) + alpha*np.random.uniform(-1,1,n_variables)
                    fireflies[i] = np.clip(fireflies[i], lower_bound, upper_bound)
                    fitnesses[i] = evaluate_solution(fireflies[i])
    best = fireflies[np.argmin(fitnesses)]
    return best, min(fitnesses)

# 15. Cuckoo Search (CS)
def cuckoo_search():
    num_nests = population_size
    nests = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_nests)]
    fitnesses = [evaluate_solution(n) for n in nests]
    pa = 0.25
    iterations = generations
    for it in range(iterations):
        for i in range(num_nests):
            step = np.random.standard_cauchy(n_variables)
            new_nest = nests[i] + step
            new_nest = np.clip(new_nest, lower_bound, upper_bound)
            f_new = evaluate_solution(new_nest)
            if f_new < fitnesses[i]:
                nests[i] = new_nest.copy()
                fitnesses[i] = f_new
        for i in range(num_nests):
            if random.random() < pa:
                nests[i] = np.random.uniform(lower_bound, upper_bound, n_variables)
                fitnesses[i] = evaluate_solution(nests[i])
    best = nests[np.argmin(fitnesses)]
    return best, min(fitnesses)

# 16. Gravitational Search Algorithm (GSA)
def gravitational_search_algorithm():
    num_agents = population_size
    agents = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_agents)]
    fitnesses = [evaluate_solution(a) for a in agents]
    velocities = [np.zeros(n_variables) for _ in range(num_agents)]
    G0 = 100; alpha_g = 20
    iterations = generations
    for it in range(iterations):
        G = G0 * exp(-alpha_g*it/iterations)
        best_fit = min(fitnesses)
        masses = [(fitnesses[i] - best_fit + 1e-9) for i in range(num_agents)]
        masses = np.array(masses) / np.sum(masses)
        for i in range(num_agents):
            force = np.zeros(n_variables)
            for j in range(num_agents):
                if i != j:
                    distance = np.linalg.norm(agents[j]-agents[i]) + 1e-9
                    force += random.random() * (G * masses[j] * (agents[j]-agents[i])) / distance
            velocities[i] = random.random()*velocities[i] + force
            agents[i] = agents[i] + velocities[i]
            agents[i] = np.clip(agents[i], lower_bound, upper_bound)
            fitnesses[i] = evaluate_solution(agents[i])
    best = agents[np.argmin(fitnesses)]
    return best, min(fitnesses)

# 17. Whale Optimization Algorithm (WOA)
def whale_optimization_algorithm():
    num_whales = population_size
    whales = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_whales)]
    fitnesses = [evaluate_solution(w) for w in whales]
    iterations = generations
    best = whales[np.argmin(fitnesses)].copy()
    for it in range(iterations):
        a = 2 - 2*(it/iterations)
        for i in range(num_whales):
            p = random.random()
            if p < 0.5:
                A = 2*a*random.random()-a
                C = 2*random.random()
                D = abs(C*best - whales[i])
                whales[i] = best - A*D
            else:
                distance_to_best = abs(best - whales[i])
                b = 1; l = random.uniform(-1,1)
                whales[i] = distance_to_best * exp(b*l) * cos(2*pi*l) + best
            whales[i] = np.clip(whales[i], lower_bound, upper_bound)
            fitnesses[i] = evaluate_solution(whales[i])
        best = whales[np.argmin(fitnesses)].copy()
    return best, min(fitnesses)

# 18. Bat Algorithm (BA)
def bat_algorithm():
    num_bats = population_size
    bats = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_bats)]
    velocities = [np.zeros(n_variables) for _ in range(num_bats)]
    freq = np.zeros(num_bats)
    loudness = np.ones(num_bats)
    pulse_rate = np.zeros(num_bats)
    fitnesses = [evaluate_solution(b) for b in bats]
    best = bats[np.argmin(fitnesses)].copy()
    iterations = generations
    for it in range(iterations):
        for i in range(num_bats):
            freq[i] = random.random()
            velocities[i] = velocities[i] + (bats[i]-best)*freq[i]
            candidate = bats[i] + velocities[i]
            candidate = np.clip(candidate, lower_bound, upper_bound)
            if random.random() > pulse_rate[i]:
                candidate = best + 0.001*np.random.randn(n_variables)
            f_candidate = evaluate_solution(candidate)
            if f_candidate <= fitnesses[i] and random.random() < loudness[i]:
                bats[i] = candidate.copy()
                fitnesses[i] = f_candidate
                loudness[i] *= 0.9
                pulse_rate[i] = pulse_rate[i] * (1 - exp(-0.9*it))
            if f_candidate < evaluate_solution(best):
                best = candidate.copy()
        for i in range(num_bats):
            bats[i] = np.clip(bats[i], lower_bound, upper_bound)
            fitnesses[i] = evaluate_solution(bats[i])
    return best, min(fitnesses)

# 19. Imperialist Competitive Algorithm (ICA)
def imperialist_competitive_algorithm():
    num_imperialists = max(2, population_size//10)
    colonies = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    fitnesses = [evaluate_solution(c) for c in colonies]
    idx_sorted = np.argsort(fitnesses)
    imperialists = [colonies[i] for i in idx_sorted[:num_imperialists]]
    empires = {i: [] for i in range(num_imperialists)}
    for i in idx_sorted[num_imperialists:]:
        empire_idx = random.randint(0, num_imperialists-1)
        empires[empire_idx].append(colonies[i])
    iterations = generations
    for it in range(iterations):
        for idx in range(num_imperialists):
            new_colonies = []
            for col in empires[idx]:
                candidate = col + np.random.uniform(-1,1,n_variables) * (imperialists[idx] - col)
                candidate = np.clip(candidate, lower_bound, upper_bound)
                new_colonies.append(candidate)
            empires[idx] = new_colonies
            candidate = imperialists[idx] + np.random.uniform(-1,1,n_variables)*0.1
            candidate = np.clip(candidate, lower_bound, upper_bound)
            if evaluate_solution(candidate) < evaluate_solution(imperialists[idx]):
                imperialists[idx] = candidate.copy()
        all_sol = imperialists.copy()
        for col_list in empires.values():
            all_sol += col_list
        best = min(all_sol, key=lambda x: evaluate_solution(x))
    return best, evaluate_solution(best)

# 20. Teaching-Learning-Based Optimization (TLBO)
def tlbo():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    best_sol = None
    best_fitness = float('inf')
    iterations = generations
    for it in range(iterations):
        fitnesses = [evaluate_solution(ind) for ind in population]
        best_index = np.argmin(fitnesses)
        teacher = population[best_index].copy()
        mean_vector = np.mean(population, axis=0)
        TF = random.choice([1,2])
        new_population = []
        for ind in population:
            r = np.random.uniform(0,1, n_variables)
            new_ind = ind + r*(teacher - TF*mean_vector)
            new_ind = np.clip(new_ind, lower_bound, upper_bound)
            if evaluate_solution(new_ind) < evaluate_solution(ind):
                new_population.append(new_ind)
            else:
                new_population.append(ind)
        population = new_population
        new_population = []
        for i in range(population_size):
            partner_idx = random.randint(0, population_size-1)
            while partner_idx == i:
                partner_idx = random.randint(0, population_size-1)
            ind = population[i]
            partner = population[partner_idx]
            r = np.random.uniform(0,1, n_variables)
            if evaluate_solution(ind) < evaluate_solution(partner):
                new_ind = ind + r*(ind - partner)
            else:
                new_ind = ind + r*(partner - ind)
            new_ind = np.clip(new_ind, lower_bound, upper_bound)
            new_population.append(new_ind)
        population = new_population
        fitnesses = [evaluate_solution(ind) for ind in population]
        current_best = min(fitnesses)
        if current_best < best_fitness:
            best_fitness = current_best
            best_sol = population[np.argmin(fitnesses)].copy()
    return best_sol, best_fitness

# 21. Cultural Algorithm (CA)
def cultural_algorithm():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    belief = np.random.uniform(lower_bound, upper_bound, n_variables)
    iterations = generations
    best_sol = population[np.argmin([evaluate_solution(p) for p in population])].copy()
    best_fitness = evaluate_solution(best_sol)
    for it in range(iterations):
        for i in range(population_size):
            population[i] = population[i] + 0.1*(belief - population[i]) + np.random.uniform(-1,1,n_variables)*0.05
            population[i] = np.clip(population[i], lower_bound, upper_bound)
        fitnesses = [evaluate_solution(p) for p in population]
        current_best = min(fitnesses)
        if current_best < best_fitness:
            best_fitness = current_best
            best_sol = population[np.argmin(fitnesses)].copy()
        belief = belief + 0.05*(best_sol - belief)
    return best_sol, best_fitness

# 22. Biogeography-Based Optimization (BBO)
def biogeography_based_optimization():
    population = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(population_size)]
    fitnesses = [evaluate_solution(p) for p in population]
    iterations = generations
    for it in range(iterations):
        immigration = 1 - np.array(fitnesses)/max(fitnesses)
        for i in range(population_size):
            candidate = population[i].copy()
            for j in range(n_variables):
                if random.random() < immigration[i]:
                    k = random.randint(0, population_size-1)
                    candidate[j] = population[k][j]
            candidate = candidate + np.random.uniform(-1,1,n_variables)*0.01
            candidate = np.clip(candidate, lower_bound, upper_bound)
            f_candidate = evaluate_solution(candidate)
            if f_candidate < fitnesses[i]:
                population[i] = candidate.copy()
                fitnesses[i] = f_candidate
        best = population[np.argmin(fitnesses)]
    return best, min(fitnesses)

# 23. Ant Lion Optimizer (ALO)
def ant_lion_optimizer():
    num_agents = population_size
    ants = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_agents)]
    antlions = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_agents)]
    fitness_antlions = [evaluate_solution(a) for a in antlions]
    iterations = generations
    for it in range(iterations):
        for i in range(num_agents):
            rand_walk = np.cumsum(np.random.uniform(-1,1,n_variables))
            candidate = ants[i] + rand_walk*0.01
            candidate = np.clip(candidate, lower_bound, upper_bound)
            if evaluate_solution(candidate) < evaluate_solution(antlions[i]):
                antlions[i] = candidate.copy()
                fitness_antlions[i] = evaluate_solution(candidate)
            ants[i] = candidate.copy()
        best_idx = np.argmin(fitness_antlions)
        best = antlions[best_idx].copy()
    return best, min(fitness_antlions)

# 24. Quantum-behaved Particle Swarm Optimization (QPSO)
def qpso():
    num_particles = population_size
    iterations = generations
    particles = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_particles)]
    fitnesses = [evaluate_solution(p) for p in particles]
    gbest = particles[np.argmin(fitnesses)].copy()
    beta = 1.0
    for it in range(iterations):
        mbest = np.mean(particles, axis=0)
        for i in range(num_particles):
            u = np.random.uniform(0,1, n_variables)
            p = particles[i]
            rand_sign = np.where(np.random.rand(n_variables) < 0.5, -1, 1)
            new_pos = p + rand_sign * beta * np.abs(mbest - p) * np.log(1/u)
            new_pos = np.clip(new_pos, lower_bound, upper_bound)
            if evaluate_solution(new_pos) < evaluate_solution(p):
                particles[i] = new_pos
        fitnesses = [evaluate_solution(p) for p in particles]
        current_best = min(fitnesses)
        if current_best < evaluate_solution(gbest):
            gbest = particles[np.argmin(fitnesses)].copy()
    return gbest, evaluate_solution(gbest)

# 25. Dragonfly Algorithm (DA)
def dragonfly_algorithm():
    num_agents = population_size
    agents = [np.random.uniform(lower_bound, upper_bound, n_variables) for _ in range(num_agents)]
    velocities = [np.zeros(n_variables) for _ in range(num_agents)]
    iterations = generations
    w = 0.5; c_s = 0.5; c_a = 0.5; c_c = 0.5; c_f = 0.5; c_e = 0.5
    food = np.random.uniform(lower_bound, upper_bound, n_variables)
    for it in range(iterations):
        for i in range(num_agents):
            S = np.random.uniform(-1,1,n_variables)
            A = np.random.uniform(-1,1,n_variables)
            C = np.random.uniform(-1,1,n_variables)
            F = food - agents[i]
            E = np.random.uniform(-1,1,n_variables)
            velocities[i] = w*velocities[i] + c_s*S + c_a*A + c_c*C + c_f*F + c_e*E
            agents[i] = agents[i] + velocities[i]
            agents[i] = np.clip(agents[i], lower_bound, upper_bound)
        food = agents[np.argmin([evaluate_solution(a) for a in agents])].copy()
    best = food.copy()
    return best, evaluate_solution(best)

###############################################
# EXECUTION OF ALL ALGORITHMS AND RESULTS
###############################################
algorithms = {
    "Genetic Algorithm": genetic_algorithm,
    "Evolution Strategy": evolution_strategy,
    "Evolutionary Programming": evolutionary_programming,
    "Simulated Annealing": simulated_annealing,
    "Tabu Search": tabu_search,
    "Ant Colony Optimization": ant_colony_optimization,
    "PSO": particle_swarm_optimization,
    "DE": differential_evolution,
    "Artificial Bee Colony (ABC)": artificial_bee_colony,
    "Variable Neighborhood Search": variable_neighborhood_search,
    "Memetic Algorithm": memetic_algorithm,
    "Scatter Search": scatter_search,
    "Harmony Search": harmony_search,
    "Firefly Algorithm": firefly_algorithm,
    "Cuckoo Search": cuckoo_search,
    "GSA": gravitational_search_algorithm,
    "WOA": whale_optimization_algorithm,
    "Bat Algorithm": bat_algorithm,
    "ICA": imperialist_competitive_algorithm,
    "TLBO": tlbo,
    "Cultural Algorithm": cultural_algorithm,
    "BBO": biogeography_based_optimization,
    "Ant Lion Optimizer": ant_lion_optimizer,
    "QPSO": qpso,
    "Dragonfly Algorithm": dragonfly_algorithm
}

results = {}
times = {}
solutions = {}
min_costs = {}

for name, func in algorithms.items():
    start = time.time()
    sol, fit = func()
    duration = time.time() - start
    results[name] = fit
    times[name] = duration
    solutions[name] = sol
    min_costs[name] = calculate_cost(sol)

result_table = pd.DataFrame({
    "Algorithm": list(results.keys()),
    "Optimal Value": [str(np.round(solutions[alg], 2).tolist()) for alg in results.keys()],
    "Minimum Cost": [np.round(min_costs[alg], 2) for alg in results.keys()],
    "Best Fitness": list(results.values()),
    "Execution Time (s)": [times[alg] for alg in results.keys()]
})

print("Comparative Table of Algorithms:")
print(result_table)

###########################################
# FUZZY LOGIC GRAPHS (RELATED)
###########################################
def fuzzy_logic_graphs():
    x_val = np.linspace(0, 20, 100)
    mf_low = fuzz.trimf(x_val, [0, 0, 10])
    plt.figure()
    plt.plot(x_val, mf_low, label="Low")
    plt.title("Individual Membership Functions")
    plt.xlabel("Universe")
    plt.ylabel("Membership")
    plt.legend()
    plt.show()

    mf_medium = fuzz.trimf(x_val, [0, 10, 20])
    mf_high = fuzz.trimf(x_val, [10, 20, 20])
    plt.figure()
    plt.plot(x_val, mf_low, label="Low")
    plt.plot(x_val, mf_medium, label="Medium")
    plt.plot(x_val, mf_high, label="High")
    plt.title("Overlapping Membership Functions")
    plt.xlabel("Universe")
    plt.ylabel("Membership")
    plt.legend()
    plt.show()

    aggregated = np.fmax(mf_low, np.fmax(mf_medium, mf_high))
    centroid = fuzz.defuzz(x_val, aggregated, 'centroid')
    plt.figure()
    plt.plot(x_val, aggregated, label="Aggregated")
    plt.axvline(x=centroid, color='red', label=f"Centroid = {centroid:.2f}")
    plt.title("Defuzzification Process")
    plt.xlabel("Universe")
    plt.ylabel("Membership")
    plt.legend()
    plt.show()

    value = 12
    degree_low = fuzz.interp_membership(x_val, mf_low, value)
    degree_medium = fuzz.interp_membership(x_val, mf_medium, value)
    degree_high = fuzz.interp_membership(x_val, mf_high, value)
    plt.figure()
    plt.plot(x_val, mf_low, label="Low")
    plt.plot(x_val, mf_medium, label="Medium")
    plt.plot(x_val, mf_high, label="High")
    plt.axvline(x=value, color='black', linestyle='--', label=f"Value = {value}")
    plt.title("Membership Degree for a Given Value")
    plt.xlabel("Universe")
    plt.ylabel("Membership")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x_val, mf_low, label="Low")
    plt.plot(x_val, mf_medium, label="Medium")
    plt.plot(x_val, mf_high, label="High")
    plt.fill_between(x_val, 0, mf_low, alpha=0.3)
    plt.fill_between(x_val, 0, mf_medium, alpha=0.3)
    plt.fill_between(x_val, 0, mf_high, alpha=0.3)
    plt.title("Universe Partition Diagram")
    plt.xlabel("Universe")
    plt.ylabel("Membership")
    plt.legend()
    plt.show()

fuzzy_logic_graphs()

# Order the result table from best to worst (lowest fitness to highest)
ordered_table = result_table.sort_values(by="Best Fitness", ascending=True)

print("\nOrdered Table (from best to worst):")
print(ordered_table)

import ast

def plot_result_table(table):
    import matplotlib.pyplot as plt

    # Bar plot for Minimum Cost per Algorithm
    plt.figure(figsize=(12, 6))
    plt.bar(table["Algorithm"], table["Minimum Cost"], color='mediumpurple')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Minimum Cost")
    plt.title("Minimum Cost per Algorithm")
    plt.tight_layout()
    plt.show()

    # Bar plot for Best Fitness per Algorithm
    plt.figure(figsize=(12, 6))
    plt.bar(table["Algorithm"], table["Best Fitness"], color='steelblue')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Best Fitness")
    plt.title("Best Fitness per Algorithm")
    plt.tight_layout()
    plt.show()

    # Bar plot for Execution Time (s) per Algorithm
    plt.figure(figsize=(12, 6))
    plt.bar(table["Algorithm"], table["Execution Time (s)"], color='seagreen')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time per Algorithm")
    plt.tight_layout()
    plt.show()

    # Plot for the Optimal Value Vectors per Algorithm
    # It is assumed that the "Optimal Value" column is a string representing a numerical list.
    plt.figure(figsize=(12, 6))
    for idx, row in table.iterrows():
        try:
            # Conversion of the string to a list of numbers
            vector = ast.literal_eval(row["Optimal Value"])
            indices_vector = range(1, len(vector) + 1)
            plt.plot(indices_vector, vector, marker='o', label=row["Algorithm"])
        except Exception as e:
            print(f"Error plotting 'Optimal Value' for {row['Algorithm']}: {e}")
    plt.xlabel("Solution Vector Index")
    plt.ylabel("Optimal Value")
    plt.title("Optimal Value Vectors per Algorithm")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Call the function to plot the result table.
plot_result_table(result_table)
