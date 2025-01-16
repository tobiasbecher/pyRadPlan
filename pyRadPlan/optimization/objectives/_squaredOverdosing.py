"""Squared Overdosing."""

from pydantic import Field

from numba import njit
from numpy import clip

from .._objective import Objective

# %% Class definition


class SquaredOverdosing(Objective):

    name = "Squared Overdosing"
    # parameter_types = ["dose"]
    parameter_names = ["d_max"]

    d_max: float = Field(default=30.0, ge=0.0)

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.d_max, self.priority)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(dose, self.d_max, self.priority)


@njit
def _compute_objective(dose, d_max, priority):

    overdose = clip(dose - d_max, a_min=0, a_max=None)

    return priority * (overdose @ overdose) / len(dose)


# @njit
def _compute_gradient(dose, d_max, priority):

    overdose = clip(dose - d_max, a_min=0, a_max=None)
    grad = 2 * overdose / len(overdose)

    return priority * grad
