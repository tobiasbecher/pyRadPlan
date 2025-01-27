from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, Bounds

from ...plan import Plan
from .._optiprob import OptimizationProblem


class FluenceOptimizationProblem(OptimizationProblem):

    penalties: ArrayLike
    result: dict[str]

    def __init__(self, pln: Union[Plan, dict] = None):
        self.penalties = np.asarray([1000, 1])
        self.target_prescription = 2.0
        self._target_voxels = None
        self._patient_voxels = None

        super().__init__(pln)

    def _initialize(self):
        super()._initialize()
        self._target_voxels = self._cst.target_union_voxels(order="numpy")
        self._patient_voxels = self._cst.patient_voxels(order="numpy")

    def _objective_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the objective functions."""
        dose = self._dij.get_result_arrays_from_intensity(x)
        target_fun = (
            np.sum((dose["physical_dose"][self._target_voxels] - self.target_prescription) ** 2)
            / self._target_voxels.size
        )
        patient_fun = (
            np.sum((dose["physical_dose"][self._patient_voxels]) ** 2) / self._patient_voxels.size
        )
        return np.array([target_fun, patient_fun])

    def _objective_function(self, x: np.ndarray) -> np.float64:
        return np.dot(np.asarray(self.penalties), self._objective_functions(x))

    def _objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective jacobian."""

        d = self._dij.get_result_arrays_from_intensity(x)

        dose_grad = np.zeros((d["physical_dose"].size, 2))
        dose_grad[self._target_voxels, 0] = (
            2
            * (d["physical_dose"][self._target_voxels] - self.target_prescription)
            / self._target_voxels.size
        )
        dose_grad[self._patient_voxels, 1] = (
            2 * (d["physical_dose"][self._patient_voxels]) / self._patient_voxels.size
        )

        return self._dij.physical_dose.flat[0].T @ dose_grad

    def _objective_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self._objective_jacobian(x) * self.penalties, axis=1)

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

    def _optimize(self):
        self.result = minimize(
            self._objective_function,
            x0=np.ones((self._dij.total_num_of_bixels,), dtype=np.float32),
            jac=self._objective_gradient,
            method="L-BFGS-B",
            options={"ftol": 1.0e-4, "maxiter": 500},
            bounds=Bounds(0, np.inf),
            # callback=callback,
        )

    def _finalize(self) -> np.ndarray:
        return self.result["x"]
