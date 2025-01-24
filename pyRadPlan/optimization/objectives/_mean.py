"""Mean dose."""
from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import ones

from .._objective import Objective, ParameterMetadata

# %% Class definition


class MeanDose(Objective):

    name = "Mean Dose"

    d_ref: Annotated[float, Field(default=60.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values):
        return _compute_objective(values, self.d_ref, self.priority)

    def compute_gradient(self, values):
        return _compute_gradient(
            values,
            self.d_ref,
            self.priority,
        )


@njit
def _compute_objective(dose, d_ref, priority):
    return priority * (dose.mean() - d_ref) ** 2


# @njit
def _compute_gradient(dose, d_ref, priority):

    grad = 2 * (dose.mean() - d_ref) * ones(dose.shape) / len(dose)
    return priority * grad
