"""Squared Underdosing."""

# %% Imports

from numba import njit
from numpy import clip, zeros

from ._objectiveClass import ObjectiveClass

# %% Class definition


class SquaredUnderdosing(ObjectiveClass):

    name = "Squared Underdosing"
    parameter_names = ["d^{min}"]
    parameter_types = ["dose"]
    parameters = 0.0
    weight = 1.0

    def __init__(self, cst, dij, dMin=parameters, weight=weight):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = SquaredUnderdosing.name
        self.parameter_names = SquaredUnderdosing.parameter_names
        self.parameter_types = SquaredUnderdosing.parameter_types
        self.parameters = dMin if isinstance(dMin, float) else float(dMin)
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(SquaredUnderdosing, SquaredUnderdosing)._check_objective(
            self,
            self.name,
            self.parameter_names,
            self.parameter_types,
            self.parameters,
            self.weight,
        )

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, struct, self.parameters, self.weight)

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
def _compute_objective(dose, struct, parameters, weight):

    underdose = clip(dose - parameters, a_min=None, a_max=0)

    return weight * (underdose @ underdose) / len(dose)


# @njit
def _compute_gradient(dose, parameters, weight, n_voxels, struct_idx):

    obj_grad = zeros((n_voxels,))
    underdose = clip(dose - parameters, a_min=None, a_max=0)
    grad = 2 * underdose / len(underdose)
    obj_grad[struct_idx] = weight * grad

    return obj_grad
