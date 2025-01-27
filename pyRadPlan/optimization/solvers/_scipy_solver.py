"""SciPy solver Class."""

from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, Bounds

from ._base_solvers import NonLinearOptimizer


class OptimizerSciPy(NonLinearOptimizer):
    """
    SciPy solver configuration class.

    This class provides methods to configure the SciPy solvers from the \
    arguments set in the treatment plan. It generates a SciPy-compatible \
    optimization problem and specifies the solver options.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_constraints : int
        Number of constraints.

    problem_instance : object of class 'WeightedSumOptimization', \
    'LexicographicOptimization', or 'ParetoOptimization'
        Instance of the optimization problem in user language.

    lower_variable_bounds : ndarray
        Lower bounds to the decision variables.

    upper_variable_bounds : ndarray
        Upper bounds to the decision variables.

    lower_constraint_bounds : ndarray
        Lower bounds to the constraints.

    upper_constraint_bounds : ndarray
        Upper bounds to the constraints.

    linear_solver : string
        Name of the linear solver to be used.

    max_iter : int
        Maximum number of iterations taken for the solver to converge.

    max_cpu_time : float
        Maximum CPU time taken for the solver to converge.

    Attributes
    ----------
    fun : object of class 'function'
        Function from the SciPy library to be called with ``arguments``.

    arguments : dict
        Dictionary with the arguments for ``fun``.
    """

    name = "SciPy minimize"
    short_name = "scipy"

    options: dict[str]
    method: Union[str, Callable]
    result: dict[str]

    def __init__(self):

        self.options = {
            "disp": False,
            "ftol": 1e-4,
        }

        self.method = "L-BFGS-B"

        self.result = None

        super().__init__()

    def solve(self, x0: ArrayLike) -> tuple[np.ndarray, dict]:
        """
        Solve a problem.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for the decision variables.

        Returns
        -------
        result : dict
        """

        self.options.update({"maxiter": self.max_iter})

        x0 = np.asarray(x0)

        bounds = Bounds(lb=self.bounds[0], ub=self.bounds[1])

        # Initialize the SciPy solution function and its arguments
        result = minimize(
            x0=x0,
            fun=self.objective,
            method=self.method,
            jac=self.gradient,
            # constraints=self.constraints,
            # hess=self.hessian,
            tol=self.abs_obj_tol,
            bounds=bounds,
            options=self.options,
        )

        return result["x"], result
