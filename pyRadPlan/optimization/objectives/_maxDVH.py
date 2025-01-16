"""Maximum DVH."""
from pydantic import Field

from numba import njit
from numpy import logical_or, quantile, sort

from .._objective import Objective


class MaxDVH(Objective):

    name = "Max DVH"
    parameter_names = ["d", "v_max"]
    # parameter_types = ["dose", "numeric"]
    # parameters = [30.0, 95.0]

    d: float = Field(default=30.0, ge=0.0)
    v_max: float = Field(default=50.0, ge=0.0, le=100.0)

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.d, self.v_max, self.priority)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(dose, self.d, self.v_max, self.priority)


@njit
def _compute_objective(dose, d, v_max, priority):

    deviation = dose - d
    dose_quantile = quantile(sort(dose)[::-1], v_max)
    mask = logical_or(dose < d, dose > dose_quantile)
    deviation[mask] = 0

    return priority * (deviation @ deviation) / len(dose)


@njit
def _compute_gradient(dose, d, v_max, priority):
    deviation = dose - d
    dose_quantile = quantile(sort(dose)[::-1], v_max)
    mask = logical_or(dose < d, dose > dose_quantile)
    deviation[mask] = 0
    return 2 * priority * deviation / len(dose)
