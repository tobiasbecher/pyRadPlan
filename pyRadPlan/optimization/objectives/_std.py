"""Dose uniformity."""

from math import sqrt
from numba import njit

from .._objective import Objective


class DoseUniformity(Objective):
    """Uniformity (minimize standard deviation) objective."""

    name = "Dose Uniformity"

    def compute_objective(self, values):
        return _compute_objective(values, self.priority)

    def compute_gradient(self, values):
        return _compute_gradient(values, self.priority)


@njit
def _compute_objective(dose, priority):

    return priority * sqrt(len(dose) / (len(dose) - 1)) * dose.std()


# @njit
def _compute_gradient(dose, priority):
    if dose.std() > 0:
        grad = (dose - dose.mean()) / (sqrt((len(dose) - 1) * len(dose)) * dose.std())
    else:
        grad = 0

    return priority * grad
