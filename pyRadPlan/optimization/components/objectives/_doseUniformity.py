"""Dose uniformity."""

# %% Imports

from math import sqrt
from numba import njit
from numpy import zeros

from ._objectiveClass import Objective

# %% Class definition


class DoseUniformity(Objective):

    name = "Dose Uniformity"
    parameter_names = []
    parameter_types = []
    parameters = []
    weight = 1.0

    def __init__(self, cst, dij, weight=weight):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = DoseUniformity.name
        self.parameter_names = DoseUniformity.parameter_names
        self.parameter_types = DoseUniformity.parameter_types
        self.parameters = DoseUniformity.parameters
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(DoseUniformity, DoseUniformity)._check_objective(
            self,
            self.name,
            self.parameter_names,
            self.parameter_types,
            self.parameters,
            self.weight,
        )

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.weight)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(
            dose, self.weight, self.dij["numOfVoxels"], self.cst[struct]["resized_indices"]
        )

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, value):
        self.parameters = value

    def get_weight(self):
        return self.weight

    def set_weight(self, value):
        self.weight = value


@njit
def _compute_objective(dose, weight):

    return weight * sqrt(len(dose) / (len(dose) - 1)) * dose.std()


# @njit
def _compute_gradient(dose, weight, n_voxels, struct_idx):

    obj_grad = zeros((n_voxels,))
    if dose.std() > 0:
        grad = (dose - dose.mean()) / (sqrt((len(dose) - 1) * len(dose)) * dose.std())
    else:
        grad = 0
    obj_grad[struct_idx] = weight * grad

    return obj_grad
