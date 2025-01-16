"""Equivalent (possibly) useless dose."""

# %% Imports

from numba import njit
from numpy import zeros, sum as npsum

from .._objective import Objective

# %% Class definition


class EUD(Objective):

    name = "EUD"
    parameter_names = ["EUD^{ref}", "k"]
    parameter_types = ["dose", "numeric"]
    parameters = [0.0, 3.5]
    weight = 1.0

    def __init__(self, cst, dij, ref=parameters[0], k=parameters[1], weight=weight):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = EUD.name
        self.parameter_names = EUD.parameter_names
        self.parameter_types = EUD.parameter_types
        self.parameters = [
            ref if isinstance(ref, float) else float(ref),
            k if isinstance(k, float) else float(k),
        ]
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(EUD, EUD)._check_objective(
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

    eud = (npsum(dose ** (1 / parameters[1])) / len(dose)) ** parameters[1]

    return weight * (eud - parameters[0]) ** 2


# njit
def _compute_gradient(dose, parameters, weight, n_voxels, struct_idx):

    eud = (npsum(dose ** (1 / parameters[1])) / len(dose)) ** parameters[1]
    eud_gradient = (
        npsum(dose ** (1 / parameters[1])) ** (parameters[1] - 1)
        * dose ** (1 / parameters[1] - 1)
        / (len(dose) ** parameters[1])
    )

    obj_grad = zeros((n_voxels,))
    grad = 2 * (eud - parameters[0]) * eud_gradient
    obj_grad[struct_idx] = weight * grad

    return obj_grad
