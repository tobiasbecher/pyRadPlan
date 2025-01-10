"""SciPy solver configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from scipy.optimize import minimize


class SciPySolver:
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

    def __init__(
        self,
        number_of_variables,
        number_of_constraints,
        problem_instance,
        lower_variable_bounds,
        upper_variable_bounds,
        lower_constraint_bounds,
        upper_constraint_bounds,
        linear_solver,
        max_iter,
        max_cpu_time,
    ):

        # Initialize the SciPy solution function and its arguments
        self.fun = minimize
        self.arguments = {
            "fun": problem_instance.objective,
            "method": linear_solver,
            "jac": problem_instance.gradient,
            "bounds": tuple(zip(lower_variable_bounds, upper_variable_bounds)),
            "tol": 1e-3,
            "options": {"maxiter": max_iter, "disp": True},
        }

    def __str__(self):
        """
        Print the class attributes.

        Returns
        -------
        string
            Class attributes as a formatted string.
        """
        return "\n".join(
            (
                "SciPySolver class attributes:",
                "----------------------------------",
                str((*self.__dict__,)),
            )
        )

    def start(self, initial_fluence):
        """
        Run the SciPy solver.

        Parameters
        ----------
        initial_fluence : ndarray
            Initialization of the fluence vector.

        Returns
        -------
        optimized_fluence : ndarray
            Optimal fluence vector.

        solver_info : dict
            Dictionary with information on the status of the algorithm, the \
            value of the constraints multipliers at the solution, and more.
        """
        result = self.fun(x0=initial_fluence, **self.arguments)

        return result.x, result.message
