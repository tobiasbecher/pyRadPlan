"""Squared Deviation."""

from pydantic import Field

from numba import njit
from numpy import zeros

from .._objective import Objective

# %% Class definition


class SquaredDeviation(Objective):

    name = "Squared Deviation"
    parameter_names = ["d_ref"]
    # parameter_types = ["dose"]

    d_ref: float = Field(default=60.0, ge=0.0)

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.d_ref, self.priority)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(dose, self.d_ref, self.priority)


@njit
def _compute_objective(dose, d_ref, priority):

    deviation = dose - d_ref

    return priority * (deviation @ deviation) / len(dose)


# @njit
def _compute_gradient(dose, d_ref, priority):

    return 2 * priority * (dose - d_ref) / len(dose)
