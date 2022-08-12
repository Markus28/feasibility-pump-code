import mip
import numpy as np
import scipy
from scipy.optimize import linprog

from feaspump.utils import get_matrix_description


def analytic_center_sparse(
    A_ub, b_ub, A_eq, b_eq, start_point=None, alpha=0.25, quad_iterations=25
):
    if start_point is None:
        start_point, r = chebyshev_center_mat(A_ub, b_ub, A_eq, b_eq)

    if A_eq is None:
        assert b_eq is None
        A_eq = np.empty((0, A_ub.shape[1]))

    y = start_point
    n = A_ub.shape[1]
    assert n == A_eq.shape[1]
    m_eq = A_eq.shape[0]

    sparse_A_ub = scipy.sparse.csr_matrix(A_ub)
    sparse_A_eq = scipy.sparse.csr_matrix(A_eq)
    quadratic = False
    n_iterations = 0
    n_quadratic_iterations = 0
    trajectory = [y]
    u = None

    while True:
        s = b_ub - sparse_A_ub @ y
        assert (s > 0).all()

        matvec = lambda v: np.concatenate(
            (
                sparse_A_ub.T @ (1 / s**2 * (sparse_A_ub @ v[:n]))
                + sparse_A_eq.T @ v[n:],
                sparse_A_eq @ v[:n],
            )
        )
        mat = scipy.sparse.linalg.LinearOperator(
            (n + m_eq, n + m_eq), matvec=matvec, rmatvec=matvec
        )

        u, info = scipy.sparse.linalg.bicgstab(
            mat, np.concatenate((sparse_A_ub.T @ (1 / s), np.zeros(m_eq))), x0=u
        )
        step = u[:n]
        if quadratic:
            y = y - step
            n_quadratic_iterations += 1
        else:
            p = np.sqrt(np.sum((sparse_A_ub @ step) / s))
            y = y - min(alpha / p, 1) * step
            quadratic = p < alpha

        trajectory.append(y)
        n_iterations += 1
        if n_quadratic_iterations > quad_iterations:
            return trajectory[-1]


def pre_dual_newton(A, s, y, alpha=0.25):
    assert (s > 0).all(), f"Slack should be positive but the min is {np.min(s)}"
    m, n = A.shape
    u = scipy.linalg.solve(
        A @ np.diag(1 / s**2) @ A.T, A @ np.diag(1 / s) @ np.ones(n), assume_a="pos"
    )
    p = np.sqrt(np.ones(n).T @ np.diag(1 / s) @ A.T @ u)
    dy = -min(alpha / p, 1) * u
    return y + dy, p < alpha


def dual_newton(A, s, y):
    m, n = A.shape
    dy = scipy.linalg.solve(
        A @ np.diag(1 / s**2) @ A.T, -A @ np.diag(1 / s) @ np.ones(n), assume_a="pos"
    )
    return y + dy


def _chebyshev_center_full(A_ub, c_ub):
    """Compute the Chebyshev center of a full-dimensional Polytope."""
    m, n = A_ub.shape

    new_A_ub = np.zeros((m, n + 1))
    new_A_ub[:, :n] = A_ub

    for i, row in enumerate(A_ub):
        assert np.linalg.norm(row) > 0
        new_A_ub[i, -1] = np.linalg.norm(row)

    f = np.zeros((n + 1,))
    f[-1] = -1
    result = linprog(
        f, A_ub=new_A_ub, b_ub=c_ub, method="highs-ipm", bounds=(None, None)
    )
    assert result.success
    x, r = result.x[:-1], result.x[-1]

    assert (
        r >= 1e-7
    ), "Tried to compute chebyshev center but polytope is not full-dimensional"

    return x, r, n


def chebyshev_center_mat(A_ub, c_ub, A_eq=None, c_eq=None, assert_nonparallel=True):
    """Compute the chebyshev center of a polytope.

    The polytope is given via inequality and equality constraints.
    """
    if A_eq is None or len(A_eq) == 0:
        assert c_eq is None or len(c_eq) == 0
        return _chebyshev_center_full(A_ub, c_ub)

    N = scipy.linalg.null_space(A_eq)

    A2 = np.ones((A_ub.shape[0], A_ub.shape[1] + 1))
    A2_eq = np.zeros((A_eq.shape[0], A_eq.shape[1] + 1))
    projection_coefficients = A_ub @ N
    A2[:, :-1] = A_ub
    assert (
        np.sqrt(np.sum(projection_coefficients**2, axis=1)) > 1e-9
    ).all() or not assert_nonparallel, "One inequality is parallel to equality"
    A2[:, -1] = np.sqrt(np.sum(projection_coefficients**2, axis=1))
    A2_eq[:, :-1] = A_eq
    f = np.zeros(A2.shape[1])
    f[-1] = -1
    sol = linprog(
        f,
        A_ub=A2,
        b_ub=c_ub,
        A_eq=A2_eq,
        b_eq=c_eq,
        method="highs-ipm",
        bounds=(None, None),
    )
    assert sol.success, sol
    center, r = sol.x[:-1], sol.x[-1]
    assert (
        r >= 1e-8
    ), "Tried to compute chebyshev center but polytope is not full-dimensional"
    print(np.min(c_ub - A_ub @ center), r)
    return center, r, N.shape[1]


def chebyshev_center(model, nonparallel_description=True):
    clone = model.copy()
    clone.verbose = 0
    clone.sense = mip.MAXIMIZE
    (A_ub, b_ub), (A_eq, b_eq), vars = get_matrix_description(clone)
    try:
        result = chebyshev_center_mat(
            A_ub, b_ub, A_eq, b_eq, assert_nonparallel=nonparallel_description
        )
        return result, {
            "A_ub": A_ub,
            "b_ub": b_ub,
            "A_eq": A_eq,
            "b_eq": b_eq,
            "vars": vars,
        }
    except AssertionError as e:
        assert (
            str(e)
            == "Tried to compute chebyshev center but polytope is not full-dimensional"
            or str(e) == "One inequality is parallel to equality"
        ), e
        is_equality = np.zeros_like(b_ub, dtype=bool)
        for i in range(len(A_ub)):
            solution = linprog(
                A_ub[i], A_ub, b_ub, A_eq, b_eq, bounds=(None, None), method="highs-ipm"
            )
            assert solution.success, "Numerical difficulties"
            is_equality[i] = solution.fun >= b_ub[i] - 1e-8

        new_A_ub = A_ub[~is_equality]
        new_b_ub = b_ub[~is_equality]
        new_A_eq = np.concatenate([A_ub[is_equality], A_eq])
        new_b_eq = np.concatenate([b_ub[is_equality], b_eq])
        return chebyshev_center_mat(
            new_A_ub, new_b_ub, new_A_eq, new_b_eq, assert_nonparallel=False
        ), {
            "A_ub": new_A_ub,
            "b_ub": new_b_ub,
            "A_eq": new_A_eq,
            "b_eq": new_b_eq,
            "vars": vars,
        }


def analytic_center(A, c, start_point=None, alpha=0.25):
    """Compute the analytic center of a full-dimensional polytope.

    The polytope is given via the inequality description Ax <= c.

    Args:
        A (np.ndarray): Constraint matrix of shape (n, m).
        c (np.ndarray): Rhs of shape (m,)
    Returns:
        list: List of points that were visited by the algorithm. The last element is the best approximation+
            to the analytic center.
    """
    if start_point is None:
        y, r, _ = chebyshev_center_mat(A, c)  # We need a feasible point
    else:
        y = start_point

    s = c - A @ y

    quadratic = False
    trajectory = [y]
    objs = [np.sum(np.log(s))]
    quad_log = [quadratic]

    n_quad_iterations = 0
    n_iterations = 0
    while n_quad_iterations < 15:
        if not quadratic:
            y, quadratic = pre_dual_newton(A.T, s, y, alpha=alpha)
            if len(objs) > 1:
                assert objs[-1] - objs[-2] >= alpha**2 - alpha**2 / (
                    2 * (1 - alpha)
                )
        else:
            y = dual_newton(A.T, s, y)
            n_quad_iterations += 1
        s = c - A @ y
        assert (s > -1e-8).all()

        objs.append(np.sum(np.log(s)))
        trajectory.append(y)
        quad_log.append(quadratic)
        n_iterations += 1

    assert (
        objs[-1] >= objs[0]
    ), f"The algorithm did not improve the potential function: {objs[0]} vs {objs[-1]}"
    return trajectory


def analytic_center_eq(A_ub, c_ub, A_eq, c_eq, x0=None):
    """Compute the analytic center of a polytope that is not full-dimensional.

    The polytope is given via the description

        A_ub * x <= c_ub
        A_eq * x  = c_eq

    We compute the nullspace of A_eq and reparametrize the problem to work in
    a full-dimensional polytope.
    """
    if x0 is None:
        x0, r, _ = chebyshev_center_mat(A_ub, c_ub, A_eq, c_eq)

    null_space = scipy.linalg.null_space(A_eq)
    A_ub_reparam = A_ub @ null_space
    c_ub_reparam = c_ub - A_ub @ x0
    reparam_analytic_center = analytic_center(
        A_ub_reparam, c_ub_reparam, start_point=np.zeros(A_ub_reparam.shape[1])
    )[-1]
    center = null_space @ reparam_analytic_center + x0
    assert np.allclose(A_eq @ center, c_eq)
    assert (A_ub @ center <= c_ub).all()
    return center
