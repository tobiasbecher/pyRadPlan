"""Dose uniformity."""

# %% Imports

from math import sqrt
from numba import njit
from numpy import zeros

from .._objective import Objective

# %% Class definition


class DoseUniformity(Objective):

    name = "Dose Uniformity"
    parameter_names = []

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.priority)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(dose, self.priority)


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
