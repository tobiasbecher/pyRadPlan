"""Squared Overdosing."""

from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import clip

from ._objective import Objective, ParameterMetadata

# %% Class definition


class SquaredOverdosing(Objective):
    """
    Squared Overdosing (piece-wise positive least-squares) objective.

    Parameters
    ----------
    d_max : float
        maximum dose value (above which we penalize)
    """

    name = "Squared Overdosing"

    d_max: Annotated[float, Field(default=30.0, ge=0.0), ParameterMetadata(kind="reference")]

    def compute_objective(self, values):
        return _compute_objective(values, self.d_max, self.priority)

    def compute_gradient(self, values):
        return _compute_gradient(values, self.d_max, self.priority)


@njit
def _compute_objective(dose, d_max, priority):

    overdose = clip(dose - d_max, a_min=0, a_max=None)

    return priority * (overdose @ overdose) / len(dose)


# @njit
def _compute_gradient(dose, d_max, priority):

    overdose = clip(dose - d_max, a_min=0, a_max=None)
    grad = 2 * overdose / len(overdose)

    return priority * grad
