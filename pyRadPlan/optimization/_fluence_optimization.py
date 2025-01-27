import numpy as np

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.plan import Plan
from pyRadPlan.dij import Dij
from pyRadPlan.stf import SteeringInformation

from .problems.optiprob_fluence import FluenceOptimizationProblem


def fluence_optimization(
    ct: CT, cst: StructureSet, stf: SteeringInformation, dij: Dij, pln: Plan
) -> np.ndarray:

    opti_prob = FluenceOptimizationProblem(pln)

    x = opti_prob.solve(ct, cst, stf, dij)

    return x
