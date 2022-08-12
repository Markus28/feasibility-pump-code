import time
from functools import partial

import mip
import numpy as np
from mip import OptimizationStatus, xsum


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
    binary_perturbation = np.where(
        w <= 0.5,
        0.5 * np.tanh(gamma * np.arctanh(2 * w)),
        1 - 0.5 * np.tanh(gamma * np.arctanh(2 * (1 - w))),
    )
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
    random_noise = np.maximum(
        np.random.uniform(-0.3, 0.7, size=rounded_binary.shape), 0
    )
    flip_indices = np.argwhere(np.abs(rounded_binary - sol_binary) + random_noise > 0.5)
    rounded_binary[flip_indices] = 1 - rounded_binary[flip_indices]
    return tuple(rounded_binary)


def check_lp_feasibility(m, rounding):
    for constr in m.constrs:
        val = (
            sum(coeff * rounding[var] for var, coeff in constr.expr.expr.items())
            + constr.expr.const
        )
        if val > 1e-9 and constr.expr.sense in ["<", "="]:
            return False
        if val < -1e-9 and constr.expr.sense in [">", "="]:
            return False
        assert constr.expr.sense in [">", "<", "="]
    return True


def feasibility_pump(
    model,
    lp_solves=3000,
    T=20,
    beta=0.5,
    exp_backoff=0,
    alpha=1,
    delta_alpha=float("inf"),
    rounding=("round_det", {}),
    do_restarts=True,
    timelimit=180,
):
    """Customizable implementation of basic feasibility pump."""
    round_fn = partial(globals()[rounding[0]], **rounding[1])

    start_time = time.time()

    clone = model.copy()
    clone.verbose = 0
    assert clone.sense == mip.MINIMIZE

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
            raise ValueError(f"Only binary programs allowed, had {variable.var_type}")

    if (3 * T) // 2 > len(binary_variable_names):
        T = int((2 / 3) * (len(binary_variables) + 0.5))
        assert (3 * T) // 2 <= len(binary_variable_names)

    objective_constraint = None

    # If `exp_backoff` is 0, we don't need to normalize for objective F-Pump
    if exp_backoff != 0:
        multiplier = np.sqrt(len(binary_variables)) / np.sqrt(
            sum(a**2 for a in clone.objective.expr.values())
        )
    else:
        multiplier = 0

    original_objective = clone.objective
    original_obj_weight = alpha

    pass_idx = 0
    n_perturbations = 0
    n_restarts = 0

    first_sol_idx = float("inf")
    first_objective = float("inf")
    incumbent_objective = float("inf")
    objective_trajectory = []
    discrepancy_trajectory = []
    solution_indices = []

    status = clone.optimize(relax=True)
    assert status == OptimizationStatus.OPTIMAL
    z_star = clone.objective_value
    sol_binary = tuple(var.x for var in binary_variables)
    rounded_binary = round_fn(sol_binary)
    last_rounded_binary = rounded_binary
    visited = {rounded_binary: alpha}

    reached_timelimit = False
    assert_next_done = False
    original_obj_weight_cache = None

    while pass_idx < lp_solves:
        assert all(val in [0, 1] for val in rounded_binary)
        assert len(rounded_binary) == len(binary_variables)
        l1_discrepancy = xsum(
            [
                var if rounded_binary[i] == 0 else 1 - var
                for i, var in enumerate(binary_variables)
            ]
        )
        clone.objective = (
            original_obj_weight * multiplier * original_objective
            + (1 - original_obj_weight) * l1_discrepancy
        )
        status = clone.optimize(relax=True)
        assert status == OptimizationStatus.OPTIMAL, status
        objective_trajectory.append(clone.objective_value)
        discrepancy_trajectory.append(l1_discrepancy.x)

        sol_binary = tuple(var.x for var in binary_variables)
        binary_satisfied = (
            np.max(np.abs(sol_binary - np.rint(sol_binary))) < 1e-6
        )  # np.allclose(sol_binary, np.rint(sol_binary))
        assert all(val in [0, 1] for val in np.rint(sol_binary))
        assert (
            binary_satisfied or l1_discrepancy.x > 1e-6
        ), f"max_diff={np.max(np.abs(sol_binary - np.rint(sol_binary)))}, assert_next_done={assert_next_done}, l1={l1_discrepancy.x}, alpha={original_obj_weight}, idx={pass_idx}, val={clone.objective_value}"
        assert (
            not assert_next_done
        ) or binary_satisfied, f"l1={l1_discrepancy.x}, alpha={original_obj_weight}, idx={pass_idx}, val={clone.objective_value}"

        if binary_satisfied:
            if assert_next_done:
                assert_next_done = False
                original_obj_weight = original_obj_weight_cache
                original_obj_weight_cache = None

            assert original_objective.x < incumbent_objective
            incumbent_objective = original_objective.x
            # incumbent_objective = min(incumbent_objective, original_objective.x)
            solution_indices.append(pass_idx)
            if first_sol_idx == float("inf"):
                first_sol_idx = pass_idx
                first_objective = original_objective.x

            if objective_constraint is not None:
                clone.remove(objective_constraint)

            objective_constraint = clone.add_constr(
                original_objective <= beta * z_star + (1 - beta) * original_objective.x
            )

        rounded_binary = round_fn(sol_binary)

        if last_rounded_binary == rounded_binary:
            rounded_binary = perturb(sol_binary, rounded_binary, T=T)
            n_perturbations += 1
        while (
            rounded_binary in visited
            and visited[rounded_binary] - original_obj_weight < delta_alpha
            and do_restarts
        ):
            rounded_binary = restart(sol_binary, rounded_binary)
            n_restarts += 1

        visited[rounded_binary] = original_obj_weight
        last_rounded_binary = rounded_binary
        original_obj_weight *= exp_backoff
        pass_idx += 1

        if (
            original_obj_weight > 1e-6
            and multiplier > 0
            or objective_constraint is not None
        ):
            rounded_dictionary = {var: var.x for var in continuous_variables}
            rounded_dictionary.update(
                {var: val for var, val in zip(binary_variables, rounded_binary)}
            )
            if check_lp_feasibility(clone, rounded_dictionary):
                original_obj_weight_cache = original_obj_weight
                original_obj_weight = 0
                assert_next_done = True

        if time.time() - start_time > timelimit:
            reached_timelimit = True
            break

    return {
        "perturbations": n_perturbations,
        "restarts": n_restarts,
        "reached_timelimit": reached_timelimit,
        "first_sol_idx": first_sol_idx,
        "best_objective": incumbent_objective,
        "first_objective": first_objective,
        "objective_trajectory": objective_trajectory,
        "discrepancy_trajectory": discrepancy_trajectory,
    }
