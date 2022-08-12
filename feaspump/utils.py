import numpy as np


def get_matrix_description(model):
    clone = model.copy()
    clone.verbose = 0
    var_names = {var.name: i for i, var in enumerate(clone.vars)}
    n_vars = len(var_names)

    bounds = {var.name: (var.lb, var.ub) for var in clone.vars}

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for constr in clone.constrs:
        row = np.zeros(n_vars)
        for var, c in constr.expr.expr.items():
            row[var_names[var.name]] = c

        if constr.expr.sense == "<":
            A_ub.append(row)
            b_ub.append(-constr.expr.const)
        elif constr.expr.sense == ">":
            A_ub.append(-row)
            b_ub.append(constr.expr.const)
        else:
            assert constr.expr.sense == "="
            A_eq.append(row)
            b_eq.append(-constr.expr.const)

    for name, bound in bounds.items():
        if bound[0] == bound[1]:
            row = np.zeros(n_vars)
            row[var_names[name]] = 1.0
            A_eq.append(row)
            b_eq.append(bound[1])
            continue

        if bound[1] < 1e4:
            row = np.zeros(n_vars)
            row[var_names[name]] = 1.0
            A_ub.append(row)
            b_ub.append(bound[1])

        if bound[0] > -1e4:
            row = np.zeros(n_vars)
            row[var_names[name]] = -1.0
            A_ub.append(row)
            b_ub.append(-bound[0])

    if len(A_eq) == 0:
        A_eq = np.empty((0, len(clone.vars)))
        b_eq = np.empty((0,))
    else:
        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)

    assert len(A_ub) > 0
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    return (A_ub, b_ub), (A_eq, b_eq), var_names
