"""Equivalent uniform dose objective."""
from typing import Annotated
from pydantic import Field

from numba import njit
from numpy import sum as npsum

from ._objective import Objective, ParameterMetadata


class EUD(Objective):
    """
    Equivalent uniform dose (EUD) objective.

    Attributes
    ----------
    k : float
        exponent.
    eud_ref : float
        reference value
    """

    name = "EUD"

    eud_ref: Annotated[float, Field(default=0.0, ge=0.0), ParameterMetadata(kind="reference")]
    k: Annotated[float, Field(default=1.0), ParameterMetadata()]

    def compute_objective(self, values):
        return _compute_objective(values, self.eud_ref, self.k, self.priority)

    def compute_gradient(self, values):
        return _compute_gradient(values, self.eud_ref, self.k, self.priority)


@njit
def _compute_objective(dose, eud_ref, eud_k, priority):

    eud = (npsum(dose ** (1 / eud_k)) / len(dose)) ** eud_k

    return priority * (eud - eud_ref) ** 2


# njit
def _compute_gradient(dose, eud_ref, eud_k, priority):

    eud = (npsum(dose ** (1 / eud_k)) / len(dose)) ** eud_k
    eud_gradient = (
        npsum(dose ** (1 / eud_k)) ** (eud_k - 1) * dose ** (1 / eud_k - 1) / (len(dose) ** eud_k)
    )

    grad = 2 * (eud - eud_ref) * eud_gradient

    return priority * grad
