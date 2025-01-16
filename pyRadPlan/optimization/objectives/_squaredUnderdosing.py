"""Squared Underdosing."""

from pydantic import Field

from numba import njit
from numpy import clip, zeros

from .._objective import Objective

# %% Class definition


class SquaredUnderdosing(Objective):

    name = "Squared Underdosing"
    # parameter_types = ["dose"]
    parameter_names = ["d_min"]

    d_min: float = Field(default=60.0, ge=0.0)

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.d_min, self.priority)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(dose, self.d_min, self.priority)


@njit
def _compute_objective(dose, d_min, priority):

    underdose = clip(dose - d_min, a_min=None, a_max=0)

    return priority * (underdose @ underdose) / len(dose)


# @njit
def _compute_gradient(dose, d_min, priority):

    underdose = clip(dose - d_min, a_min=None, a_max=0)
    grad = 2 * underdose / len(underdose)

    return priority * grad
