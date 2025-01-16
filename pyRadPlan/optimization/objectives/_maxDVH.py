"""Maximum DVH."""

# %% Imports

from numba import njit
from numpy import logical_or, quantile, sort, zeros

from .._objective import Objective

# %% Class definition


class MaxDVH(Objective):

    name = "Max DVH"
    parameter_names = ["d", "V^{max}"]
    parameter_types = ["dose", "numeric"]
    parameters = [30.0, 95.0]
    weight = 1.0

    def __init__(self, cst, dij, d=parameters[0], vMax=parameters[1], weight=weight):

        self.cst = cst
        self.dij = dij

        self.adjusted_params = False

        self.name = MaxDVH.name
        self.parameter_names = MaxDVH.parameter_names
        self.parameter_types = MaxDVH.parameter_types
        self.parameters = [
            d if isinstance(d, float) else float(d),
            vMax if isinstance(vMax, float) else float(vMax),
        ]
        self.weight = weight if isinstance(weight, float) else float(weight)

        super(MaxDVH, MaxDVH)._check_objective(
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

    deviation = dose - parameters[0]
    dose_quantile = quantile(sort(dose)[::-1], parameters[1])
    mask = logical_or(dose < parameters[0], dose > dose_quantile)
    deviation[mask] = 0

    return weight * (deviation @ deviation) / len(dose)


@njit
def _compute_gradient(dose, parameters, weight, n_voxels, struct_idx):

    deviation = dose - parameters[0]
    dose_quantile = quantile(sort(dose)[::-1], parameters[1])
    mask = logical_or(dose < parameters[0], dose > dose_quantile)
    deviation[mask] = 0
    obj_grad = zeros(n_voxels)
    obj_grad[struct_idx] = 2 * weight * deviation / len(dose)

    return obj_grad
