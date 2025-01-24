"""Squared Deviation."""

from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import zeros

from .._objective import Objective, ParameterMetadata

# %% Class definition


class SquaredDeviation(Objective):

    name = "Squared Deviation"

    d_ref: Annotated[float, Field(default=60.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values):
        return _compute_objective(values, self.d_ref, self.priority)

    def compute_gradient(self, values):
        return _compute_gradient(values, self.d_ref, self.priority)


@njit
def _compute_objective(dose, d_ref, priority):

    deviation = dose - d_ref

    return priority * (deviation @ deviation) / len(dose)


# @njit
def _compute_gradient(dose, d_ref, priority):

    return 2 * priority * (dose - d_ref) / len(dose)
