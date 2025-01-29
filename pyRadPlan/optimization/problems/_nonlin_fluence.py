from typing import Union, cast
import numpy as np
from numpy.typing import ArrayLike


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

    def _objective_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the objective functions."""

        q_vectors = {}
        q_scenarios = {}

        # Check & get Caches
        for q in self._quantities:
            q_vectors[q.identifier] = q.compute(x)
            q_scenarios[q.identifier] = q.scenarios

        # Loop over all objectives
        f_vals = []

        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                f_vals.append(
                    sum(
                        [
                            obj.compute_objective(q_vectors[obj.quantity].flat[scen_ix][ix])
                            for scen_ix in q_scenarios[obj.quantity]
                        ]
                    )
                )

        # return as numpy array
        return np.asarray(f_vals, dtype=np.float64)

    def _objective_function(self, x: np.ndarray) -> np.float64:
        return np.sum(self._objective_functions(x))
        # return np.dot(np.asarray(self.penalties), self._objective_functions(x))

    def _objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective jacobian."""

        q_vectors = {}
        q_scenarios = {}
        if self._grad_cache_intermediate is None:
            initialize_cache = True
            self._grad_cache_intermediate = {}
        else:
            initialize_cache = False

        # Check & get Caches
        for q in self._quantities:
            q_vectors[q.identifier] = q.compute(x)
            q_scenarios[q.identifier] = q.scenarios
            if initialize_cache:
                self._grad_cache_intermediate[q.identifier] = np.zeros(
                    (
                        len(self._objectives_per_quantity[q.identifier]),
                        self._dij.dose_grid.num_voxels,
                    ),
                    dtype=np.float32,
                )
            else:
                self._grad_cache_intermediate[q.identifier].fill(0.0)

        cnt = 0
        for obj_info in self._objective_list:
            ix = obj_info[0]
            tmp_obj_list = cast(list[Objective], obj_info[1])
            for obj in tmp_obj_list:
                q_cache_index = self._q_cache_index[cnt]
                for scen_ix in q_scenarios[obj.quantity]:
                    self._grad_cache_intermediate[obj.quantity][
                        q_cache_index, ix
                    ] += obj.compute_gradient(q_vectors[obj.quantity].flat[scen_ix][ix])
                cnt += 1

        # perform chain rule and store in cache
        if self._grad_cache is None:
            self._grad_cache = np.zeros((cnt, self._dij.total_num_of_bixels), dtype=np.float64)
        else:
            self._grad_cache.fill(0.0)

        for q in self._quantities:
            for scen_ix in q_scenarios[q.identifier]:
                self._grad_cache[
                    self._objectives_per_quantity[q.identifier], :
                ] += q.compute_chain_derivative(
                    self._grad_cache_intermediate[q.identifier], x
                ).flat[
                    scen_ix
                ]

        return self._grad_cache

    def _objective_gradient(self, x: np.ndarray) -> np.ndarray:
        # return np.sum(self._objective_jacobian(x) * self.penalties, axis=1)
        return np.sum(self._objective_jacobian(x), axis=0)

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
