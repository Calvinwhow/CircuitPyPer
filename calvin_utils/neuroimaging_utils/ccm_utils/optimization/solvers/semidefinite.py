"""CVXPY solver for a supplied semidefinite target and constraints."""

import numpy as np
from time import perf_counter

try:
    import cvxpy as cp
except ImportError:
    cp = None


def main(target, constraint, shape, *, solver_name="SCS", max_iter=20_000, tolerance=1e-5):
    """Solve a supplied semidefinite maximization problem.
    target: callable; CVXPY scalar objective receiving the matrix variable and norm operators.
    constraint: callable; constraint list receiving the matrix variable and trace operator.
    shape: tuple[int, int]; square shape (B, B) of the lifted matrix.
    solver_name: str; installed CVXPY conic solver name.
    max_iter: int; maximum conic-solver iterations.
    tolerance: float; solver feasibility and optimality tolerance.
    """
    if cp is None:
        raise ImportError("Sparse semidefinite optimization requires CVXPY.")
    theta = cp.Variable(shape, symmetric=True)
    problem = cp.Problem(cp.Maximize(target(theta, cp.trace, cp.norm1, cp.sum_squares)), constraint(theta, cp.trace))
    options = {"eps": tolerance, "max_iters": max_iter} if solver_name.upper() == "SCS" else {"tol_gap_abs": tolerance, "tol_gap_rel": tolerance, "tol_feas": tolerance, "max_iter": max_iter} if solver_name.upper() == "CLARABEL" else {}
    started = perf_counter()
    problem.solve(solver=solver_name.upper(), **options)
    elapsed = perf_counter() - started
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Sparse SDP failed with status {problem.status!r}.")
    return {"parameters": np.asarray(theta.value, dtype=float), "objective": float(problem.value), "status": str(problem.status), "solver": solver_name.upper(), "solve_time": float(elapsed), "n_iterations": problem.solver_stats.num_iters}
