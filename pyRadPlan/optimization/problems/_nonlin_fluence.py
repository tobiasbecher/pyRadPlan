from typing import Union, cast
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, Bounds

from ...plan import Plan
from .._optiprob import NonLinearPlanningProblem
from ..solvers import NonLinearOptimizer
from .._objective import Objective


class NonLinearFluencePlanningProblem(NonLinearPlanningProblem):

    penalties: ArrayLike
    result: dict[str]

    def __init__(self, pln: Union[Plan, dict] = None):
        self.penalties = np.asarray([1000, 1])
        self.target_prescription = 2.0
        self._target_voxels = None
        self._patient_voxels = None

        self._grad_cache_intermediate = None
        self._grad_cache = None
        self._result_cache = None
        self._w_cache = None

        super().__init__(pln)

    def _initialize(self):
        super()._initialize()

        # Check if the solver is adequate to solve this problem
        # TODO: check that it can do constraints
        if not isinstance(self.solver, NonLinearOptimizer):
            raise ValueError("Solver must be an instance of SolverBase")

        self.solver.objective = self._objective_function
        self.solver.gradient = self._objective_gradient
        self.solver.bounds = (0.0, np.inf)
        self.solver.max_iter = 500
        self.solver.options = {
            "disp": True,
            "ftol": 1e-4,
        }

    def _objective_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the objective functions."""

        # Check & get Caches
        if not np.array_equal(x, self._w_cache):
            self._result_cache = self._dij.get_result_arrays_from_intensity(x)
            self._w_cache = x.copy()

        dose = self._result_cache["physical_dose"]

        # Loop over all objectives
        f_vals = []

        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            f_vals += [obj.compute_objective(dose[ix]) for obj in tmp_obj_list]

        # return as numpy array
        return np.asarray(f_vals, dtype=np.float64)

    def _objective_function(self, x: np.ndarray) -> np.float64:
        return np.sum(self._objective_functions(x))
        # return np.dot(np.asarray(self.penalties), self._objective_functions(x))

    def _objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective jacobian."""

        # Check & get caaches
        if not np.array_equal(x, self._w_cache):
            self._result_cache = self._dij.get_result_arrays_from_intensity(x)
            self._w_cache = x.copy()

        dose = self._result_cache["physical_dose"]

        if self._grad_cache_intermediate is None:
            num_objectives = sum([len(obj_info[1]) for obj_info in self._objective_list])
            self._grad_cache_intermediate = np.zeros((num_objectives, dose.size), dtype=np.float32)

        # Loop over objectives
        cnt = 0
        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                self._grad_cache_intermediate[cnt, ix] = obj.compute_gradient(dose[ix])
                cnt += 1

        # perform chain rule and store in cache
        self._grad_cache = self._dij.physical_dose.flat[0].T @ self._grad_cache_intermediate.T

        return self._grad_cache

    def _objective_gradient(self, x: np.ndarray) -> np.ndarray:
        # return np.sum(self._objective_jacobian(x) * self.penalties, axis=1)
        return np.sum(self._objective_jacobian(x), axis=1)

    def _objective_hessian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective hessian."""
        return {}

    def _constraint_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint functions."""
        return None

    def _constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint jacobian."""
        return None

    def _constraint_jacobian_structure(self) -> np.ndarray:
        """Define the constraint jacobian structure."""
        return None

    def _variable_bounds(self, x: np.ndarray) -> np.ndarray:
        """Define the variable bounds."""
        return {}

    def _solve(self) -> tuple[np.ndarray, dict]:
        """Solve the problem."""

        x0 = np.zeros((self._dij.total_num_of_bixels,), dtype=np.float64)
        return self.solver.solve(x0)
