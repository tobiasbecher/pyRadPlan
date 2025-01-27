import numpy as np

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.plan import Plan
from pyRadPlan.dij import Dij
from pyRadPlan.stf import SteeringInformation

from .problems import NonLinearFluencePlanningProblem


def fluence_optimization(
    ct: CT, cst: StructureSet, stf: SteeringInformation, dij: Dij, pln: Plan
) -> np.ndarray:

    opti_prob = NonLinearFluencePlanningProblem(pln)

    x, _result_info = opti_prob.solve(ct, cst, stf, dij)

    return x
