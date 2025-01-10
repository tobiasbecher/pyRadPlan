"""Squared Deviation."""

# %% Imports

from numba import njit
from numpy import zeros

from ._objectiveClass import ObjectiveClass

# %% Class definition


class SquaredDeviation(ObjectiveClass):

    name = "Squared Deviation"
    parameter_names = ["d^{ref}"]
    parameter_types = ["dose"]
    parameters = 30.0
    weight = 1.0

    def __init__(self, cst, dij, dRef=parameters, weight=weight):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = SquaredDeviation.name
        self.parameter_names = SquaredDeviation.parameter_names
        self.parameter_types = SquaredDeviation.parameter_types
        self.parameters = dRef if isinstance(dRef, float) else float(dRef)
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(SquaredDeviation, SquaredDeviation)._check_objective(
            self,
            self.name,
            self.parameter_names,
            self.parameter_types,
            self.parameters,
            self.weight,
        )

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.parameters, self.weight)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(
            dose,
            self.parameters,
            self.weight,
            self.dij["numOfVoxels"],
            self.cst[struct]["resized_indices"],
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
def _compute_objective(dose, parameters, weight):

    deviation = dose - parameters

    return weight * (deviation @ deviation) / len(dose)


# @njit
def _compute_gradient(dose, parameters, weight, n_voxels, struct_idx):

    obj_grad = zeros(n_voxels)
    obj_grad[struct_idx] = 2 * weight * (dose - parameters) / len(dose)

    return obj_grad
