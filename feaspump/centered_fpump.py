import mip
from mip import OptimizationStatus, xsum
import numpy as np
import time
from benchmarking.fpump.rounding import FischettiRounder
from functools import partial
from feaspump.utils import get_matrix

def round_det(arr):
    return tuple(round(x) for x in arr)

def round_rand(sol_binary):
    sol_binary = np.array(sol_binary)
    w = np.random.rand(*sol_binary.shape)
    binary_perturbation = np.where(w <= 0.5, 2 * w * (1 - w), 1 - 2 * w * (1 - w))
    return tuple((sol_binary + binary_perturbation).astype(int))

def round_tanh(sol_binary, gamma=2):
    sol_binary = np.array(sol_binary)
    w = np.random.rand(*sol_binary.shape)
    binary_perturbation = np.where(w <= 0.5, 0.5 * np.tanh(gamma * np.arctanh(2 * w)), 1 - 0.5 * np.tanh(gamma * np.arctanh(2 * (1-w))))
    return tuple((sol_binary + binary_perturbation).astype(int))

def perturb(sol_binary, rounded_binary, T=20):
    sol_binary = np.array(sol_binary)
    rounded_binary = np.array(rounded_binary)
    discrepancy_binary = np.abs(rounded_binary - sol_binary)
    number_flips = np.random.randint(T // 2, (3 * T) // 2)
    flip_indices = np.argpartition(discrepancy_binary, -number_flips)[-number_flips:]
    rounded_binary[flip_indices] = 1 - rounded_binary[flip_indices]
    return tuple(rounded_binary)

def restart(sol_binary, rounded_binary):
    sol_binary = np.array(sol_binary)
    rounded_binary = np.array(rounded_binary)
    random_noise = np.maximum(np.random.uniform(-0.3, 0.7, size=rounded_binary.shape), 0)
    flip_indices = np.argwhere(np.abs(rounded_binary - sol_binary) + random_noise > 0.5)
    rounded_binary[flip_indices] = 1 - rounded_binary[flip_indices]
    return tuple(rounded_binary)

def feasibility_pump(model, center, lp_solves=3000, T=20, exp_backoff=0, alpha=1, delta_alpha=float("inf"), do_restarts=True, timelimit=600, n_samples=10, use_min_norm=False):
    """Customizable implementation of basic feasibility pump."""
    start_time = time.time()

    clone = model.copy()
    clone.verbose = 0
    assert clone.sense == mip.MINIMIZE
    (A_ub, b_ub), (A_eq, b_eq), var_names = get_matrix_description(clone)

    binary_variables = []
    continuous_variables = []
    binary_variable_names = []
    continuous_variable_names = []

    # Keep track of which variables are binary, general
    for variable in clone.vars:
        if variable.var_type == mip.CONTINUOUS:
            continuous_variables.append(variable)
            continuous_variable_names.append(variable.name)
        elif variable.var_type == mip.BINARY:
            binary_variables.append(variable)
            binary_variable_names.append(variable.name)
        else:
            raise ValueError("Only binary programs allowed")

    n_variables = len(binary_variables) + len(continuous_variables)
    assert n_variables == len(var_names)

    if (3 * T) // 2 > len(binary_variable_names):
        T = int((2 / 3) * (len(binary_variables) + 0.5))
        assert (3 * T) // 2 <= len(binary_variable_names)

    objective_constraint = None

    # If `exp_backoff` is 0, we don't need to normalize for objective F-Pump
    if exp_backoff != 0:
        multiplier = np.sqrt(len(binary_variables)) / np.sqrt(sum(a ** 2 for a in clone.objective.expr.values()))
    else:
        multiplier = 0

    original_objective = clone.objective
    original_obj_weight = alpha

    pass_idx = 0
    n_perturbations = 0
    n_restarts = 0

    first_sol_idx = float("inf")
    objective_trajectory = []
    discrepancy_trajectory = []
    status = clone.optimize(relax=True)
    assert status == OptimizationStatus.OPTIMAL

    np_solution = np.zeros(n_variables)
    np_center = np.zeros(n_variables)
    binary_indices = []
    for var in binary_variables:
        np_solution[var_names[var.name]] = var.x
        np_center[var_names[var.name]] = center[var.name]
        binary_indices.append(var_names[var.name])
    for var in continuous_variables:
        np_solution[var_names[var.name]] = var.x
        np_center[var_names[var.name]] = center[var.name]

    assert abs(sum(coeff * np_solution[var_names[var.name]] for var, coeff in original_objective.expr.items()) + original_objective.const - original_objective.x) < 1e-8, "Can't compute objective correctly"
    assert np.allclose(A_eq @ np_center, b_eq) and (A_ub @ np_center <= b_ub + 1e-8).all()

    reached_timelimit = False

    incumbent_norm = float("inf")
    incumbent_rounding = None
    for gamma in np.linspace(0, 1, n_samples)[:-1]:
        interpolation = gamma * np_center + (1 - gamma) * np_solution
        rounded = np.copy(interpolation)
        rounded[binary_indices] = np.rint(rounded[binary_indices])
        assert np.logical_or(rounded[binary_indices] == 0, rounded[binary_indices] == 1).all()
        if incumbent_rounding is None:
            incumbent_rounding = rounded

        inequality_satisfied = ((A_ub @ rounded) <= b_ub + 1e-8).all()
        equality_satisfied = np.allclose(A_eq @ rounded, b_eq)
        infinity_norm = np.max(np.abs(rounded - interpolation))
        if infinity_norm < incumbent_norm:
            if use_min_norm:
                incumbent_rounding = rounded
            incumbent_norm = infinity_norm

        if inequality_satisfied and equality_satisfied:
            objective_val = sum(coeff * rounded[var_names[var.name]] for var, coeff in original_objective.expr.items()) + original_objective.const
            first_sol_idx = pass_idx
            return {"perturbations": n_perturbations,
                "restarts": n_restarts, "reached_timelimit": reached_timelimit, "first_sol_idx": first_sol_idx,
                "first_objective": objective_val}



    rounded_binary = tuple(incumbent_rounding[var_names[name]] for name in binary_variable_names)
    last_rounded_binary = rounded_binary
    visited = {rounded_binary: alpha}

    reached_timelimit = False

    while pass_idx < lp_solves:
        l1_discrepancy = xsum([var if rounded_binary[i] == 0 else 1 - var for i, var in enumerate(binary_variables)])
        clone.objective = original_obj_weight * multiplier * original_objective \
                          + (1 - original_obj_weight) * l1_discrepancy
        status = clone.optimize(relax=True)
        assert status == OptimizationStatus.OPTIMAL, status
        objective_trajectory.append(clone.objective_value)
        discrepancy_trajectory.append(l1_discrepancy.x)

        # Assemble numpy vector
        np_solution = np.zeros(n_variables)
        for var in binary_variables:
            np_solution[var_names[var.name]] = var.x
        for var in continuous_variables:
            np_solution[var_names[var.name]] = var.x

        # Walk towards center
        incumbent_norm = float("inf")
        incumbent_rounding = None
        incumbent_interpolation = None
        for gamma in np.linspace(0, 1, n_samples)[:-1]:
            interpolation = gamma * np_center + (1 - gamma) * np_solution
            rounded = np.copy(interpolation)
            rounded[binary_indices] = np.rint(rounded[binary_indices])
            assert np.logical_or(rounded[binary_indices] == 0, rounded[binary_indices] == 1).all()

            inequality_satisfied = ((A_ub @ rounded) <= b_ub + 1e-8).all()
            equality_satisfied = np.allclose(A_eq @ rounded, b_eq)
            infinity_norm = np.max(np.abs(rounded - interpolation))
            if infinity_norm < incumbent_norm:
                rounding_tuple = tuple(rounded[var_names[name]] for name in binary_variable_names)
                if (use_min_norm and (not do_restarts or rounding_tuple not in visited or visited[rounding_tuple] - original_obj_weight > delta_alpha) and (not do_restarts or rounding_tuple != last_rounded_binary)) or incumbent_rounding is None:
                    incumbent_interpolation = tuple(interpolation[var_names[name]] for name in binary_variable_names)
                    incumbent_rounding = tuple(rounded[var_names[name]] for name in binary_variable_names)
                    incumbent_norm = infinity_norm

            if inequality_satisfied and equality_satisfied:
                objective_val = sum(coeff * rounded[var_names[var.name]] for var, coeff in
                                    original_objective.expr.items()) + original_objective.const
                first_sol_idx = pass_idx
                return {"perturbations": n_perturbations,
                        "restarts": n_restarts, "reached_timelimit": reached_timelimit, "first_sol_idx": first_sol_idx,
                        "first_objective": objective_val}

        sol_binary = incumbent_interpolation
        rounded_binary = incumbent_rounding

        if last_rounded_binary == rounded_binary:
            rounded_binary = perturb(sol_binary, rounded_binary, T=T)
            n_perturbations += 1
        while rounded_binary in visited and visited[rounded_binary] - original_obj_weight < delta_alpha and do_restarts:
            rounded_binary = restart(sol_binary, rounded_binary)
            n_restarts += 1


        visited[rounded_binary] = original_obj_weight
        last_rounded_binary = rounded_binary
        original_obj_weight *= exp_backoff
        pass_idx += 1

        if time.time() - start_time > timelimit:
            reached_timelimit = True
            break

    return {"perturbations": n_perturbations,
            "restarts": n_restarts, "reached_timelimit": reached_timelimit, "first_sol_idx": first_sol_idx, "first_objective": float("inf"),}