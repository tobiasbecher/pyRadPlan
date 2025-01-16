"""Equivalent (possibly) useless dose."""
from pydantic import Field

from numba import njit
from numpy import zeros, sum as npsum

from .._objective import Objective


class EUD(Objective):

    name = "EUD"
    parameter_names = ["EUD_ref", "k"]
    k: float = Field(default=1.0)
    EUD_ref: float = Field(default=0.0, ge=0.0)

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.EUD_ref, self.k, self.priority)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(dose, self.EUD_ref, self.k, self.priority)


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
