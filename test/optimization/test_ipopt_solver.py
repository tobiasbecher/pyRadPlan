import pytest
import numpy as np

from pyRadPlan.optimization.solvers import get_solver, OptimizerIpopt, SolverBase

if OptimizerIpopt is None:
    pytest.skip("IPOPT not installed", allow_module_level=True)


def test_get_solver_ipopt():
    solver = get_solver("ipopt")
    assert isinstance(solver, OptimizerIpopt)
    assert isinstance(solver, SolverBase)
    assert solver.short_name == "ipopt"


def test_simple_problem_ipopt():
    solver = get_solver("ipopt")
    # Define the problem
    def objective(x):
        return x[0] ** 2 + x[1] ** 2

    def gradient(x):
        return [2 * x[0], 2 * x[1]]

    solver.objective = objective
    solver.gradient = gradient

    # Initial guess
    x0 = [1.0, 1.0]

    # Solve
    result = solver.solve(x0)

    assert np.all(np.isclose(result[0], 0.0))
