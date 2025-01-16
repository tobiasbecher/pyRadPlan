"""Mean dose."""

# %% Imports

from numba import njit
from numpy import ones, zeros

from .._objective import Objective

# %% Class definition


class MeanDose(Objective):

    name = "Mean Dose"
    parameter_names = ["d^{ref}"]
    parameter_types = ["dose"]
    parameters = 0.0
    priority = 1.0

    def __init__(self, cst, dij, dRef=parameters, priority=priority):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = MeanDose.name
        self.parameter_names = MeanDose.parameter_names
        self.parameter_types = MeanDose.parameter_types
        self.parameters = dRef if isinstance(dRef, float) else float(dRef)
        self.priority = priority if isinstance(priority, float) else float(priority)

        super(MeanDose, MeanDose)._check_objective(
            self,
            self.name,
            self.parameter_names,
            self.parameter_types,
            self.parameters,
            self.priority,
        )

    def compute_objective(self, dose, struct):
        return _compute_objective(dose, self.parameters, self.priority)

    def compute_gradient(self, dose, struct):
        return _compute_gradient(
            dose,
            self.parameters,
            self.priority,
            self.dij["numOfVoxels"],
            self.cst[struct]["resized_indices"],
        )

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, value):
        self.parameters = value

    def get_priority(self):
        return self.priority

    def set_priority(self, value):
        self.priority = value


@njit
def _compute_objective(dose, parameters, priority):
    return priority * (dose.mean() - parameters) ** 2


# @njit
def _compute_gradient(dose, parameters, priority, n_voxels, struct_idx):

    obj_grad = zeros((n_voxels,))
    grad = 2 * (dose.mean() - parameters) * ones(dose.shape) / len(dose)
    obj_grad[struct_idx] = priority * grad

    return obj_grad
