from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.plan import Plan
from pyRadPlan.dij import Dij
from pyRadPlan.stf import SteeringInformation

import numpy as np
from scipy.optimize import minimize, Bounds


def fluence_optimization(
    ct: CT, cst: StructureSet, stf: SteeringInformation, dij: Dij, pln: Plan
) -> np.ndarray:
    resampled_ct = ct.resample_to_grid(dij.dose_grid)
    resampled_cst = cst.resample_on_new_ct(resampled_ct)

    target_voxels = resampled_cst.target_union_voxels(order="numpy")
    patient_voxels = resampled_cst.patient_voxels(order="numpy")

    def objective_function(x: np.ndarray):
        dose = dij.get_result_arrays_from_intensity(x)
        target_dose = (
            np.sum((dose["physical_dose"][target_voxels] - 2.0) ** 2) / target_voxels.size
        )
        patient_dose = np.sum((dose["physical_dose"][patient_voxels]) ** 2) / target_voxels.size

        return 1000 * target_dose + patient_dose

    def objective_gradient_dose(d: dict[str, np.ndarray]):
        dose_grad = np.zeros_like(d["physical_dose"])
        target_dose = 2 * (d["physical_dose"][target_voxels] - 2.0) / target_voxels.size
        patient_dose = 2 * (d["physical_dose"][patient_voxels]) / target_voxels.size
        dose_grad[target_voxels] += 1000 * target_dose
        dose_grad[patient_voxels] += patient_dose

        return dose_grad

    def gradient_chainrule(x: np.ndarray):
        dose = dij.get_result_arrays_from_intensity(x)
        dose_grad = objective_gradient_dose(dose)
        w_grad = dij.physical_dose.flat[0].T @ dose_grad
        return w_grad

    def callback(intermediate_result) -> None:
        print(intermediate_result)

    result = minimize(
        objective_function,
        x0=np.ones((dij.total_num_of_bixels,), dtype=np.float32),
        jac=gradient_chainrule,
        method="L-BFGS-B",
        options={"ftol": 1.0e-4, "maxiter": 500},
        bounds=Bounds(0, np.inf),
        # callback=callback,
    )

    return result.x
