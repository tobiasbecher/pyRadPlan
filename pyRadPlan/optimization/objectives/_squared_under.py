"""Squared Underdosing."""

from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import clip, zeros

from .._objective import Objective, ParameterMetadata

# %% Class definition


class SquaredUnderdosing(Objective):

    name = "Squared Underdosing"

    d_min: Annotated[float, Field(default=60.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values):
        return _compute_objective(values, self.d_min, self.priority)

    def compute_gradient(self, values):
        return _compute_gradient(values, self.d_min, self.priority)


@njit
def _compute_objective(dose, d_min, priority):

    underdose = clip(dose - d_min, a_min=None, a_max=0)

    return priority * (underdose @ underdose) / len(dose)


# @njit
def _compute_gradient(dose, d_min, priority):

    underdose = clip(dose - d_min, a_min=None, a_max=0)
    grad = 2 * underdose / len(underdose)

    return priority * grad
