"""Mean dose."""

# %% Imports

from numba import njit
from numpy import ones, zeros

from ._objectiveClass import ObjectiveClass

# %% Class definition


class MeanDose(ObjectiveClass):

    name = "Mean Dose"
    parameter_names = ["d^{ref}"]
    parameter_types = ["dose"]
    parameters = 0.0
    weight = 1.0

    def __init__(self, cst, dij, dRef=parameters, weight=weight):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = MeanDose.name
        self.parameter_names = MeanDose.parameter_names
        self.parameter_types = MeanDose.parameter_types
        self.parameters = dRef if isinstance(dRef, float) else float(dRef)
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(MeanDose, MeanDose)._check_objective(
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
    return weight * (dose.mean() - parameters) ** 2


# @njit
def _compute_gradient(dose, parameters, weight, n_voxels, struct_idx):

    obj_grad = zeros((n_voxels,))
    grad = 2 * (dose.mean() - parameters) * ones(dose.shape) / len(dose)
    obj_grad[struct_idx] = weight * grad

    return obj_grad
