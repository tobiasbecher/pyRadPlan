"""Squared Overdosing."""

# %% Imports

from numba import njit
from numpy import clip, zeros

from ._objectiveClass import Objective

# %% Class definition


class SquaredOverdosing(Objective):

    name = "Squared Overdosing"
    parameter_names = ["d^{max}"]
    parameter_types = ["dose"]
    parameters = 100000.0
    weight = 1.0

    def __init__(self, cst, dij, dMax=parameters, weight=weight):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = SquaredOverdosing.name
        self.parameter_names = SquaredOverdosing.parameter_names
        self.parameter_types = SquaredOverdosing.parameter_types
        self.parameters = dMax if isinstance(dMax, float) else float(dMax)
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(SquaredOverdosing, SquaredOverdosing)._check_objective(
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

    overdose = clip(dose - parameters, a_min=0, a_max=None)

    return weight * (overdose @ overdose) / len(dose)


# @njit
def _compute_gradient(dose, parameters, weight, n_voxels, struct_idx):

    obj_grad = zeros((n_voxels,))
    overdose = clip(dose - parameters, a_min=0, a_max=None)
    grad = 2 * overdose / len(overdose)
    obj_grad[struct_idx] = weight * grad

    return obj_grad
