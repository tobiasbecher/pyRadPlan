"""Maximum DVH."""
from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import logical_or, quantile, sort

from .._objective import Objective, ParameterMetadata


class MaxDVH(Objective):

    name = "Max DVH"

    d: Annotated[float, Field(default=30.0, ge=0.0), ParameterMetadata(kind="reference")]
    v_max: Annotated[
        float, Field(default=50.0, ge=0.0, le=100.0), ParameterMetadata(kind="relative_volume")
    ]

    def compute_objective(self, values):
        return _compute_objective(values, self.d, self.v_max, self.priority)

    def compute_gradient(self, values):
        return _compute_gradient(values, self.d, self.v_max, self.priority)


@njit
def _compute_objective(dose, d, v_max, priority):

    deviation = dose - d
    dose_quantile = quantile(sort(dose)[::-1], v_max / 100.0)
    mask = logical_or(dose < d, dose > dose_quantile)
    deviation[mask] = 0

    return priority * (deviation @ deviation) / len(dose)


@njit
def _compute_gradient(dose, d, v_max, priority):
    deviation = dose - d
    dose_quantile = quantile(sort(dose)[::-1], v_max / 100.0)
    mask = logical_or(dose < d, dose > dose_quantile)
    deviation[mask] = 0
    return 2 * priority * deviation / len(dose)
