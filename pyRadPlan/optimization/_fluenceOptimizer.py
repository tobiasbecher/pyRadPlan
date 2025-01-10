#%% External package import

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from math import inf
from numpy import (
    exp,
    hstack,
    ones,
    setdiff1d,
    union1d,
)
from scipy.ndimage import zoom
from time import time

from .projections import ConstantRBEProjection, DoseProjection
from .components import OptimizationProblem
from .solvers._scipy_solver import SciPySolver

from pyRadPlan.plan import validate_pln
from pyRadPlan.dij import Dij
from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet


class FluenceOptimizer:
    def flatten(iterable):

        for it in iterable:
            if isinstance(it, Iterable) and not isinstance(it, (str, bytes)):
                yield from FluenceOptimizer.flatten(it)
            else:
                yield it

    def sigmoid(x, A, B):
        return (
            1 / (1 + exp(-A * x + B))
            if not isinstance(x, (tuple, list))
            else tuple(1 / (1 + exp(-A * xi + B)) for xi in x)
        )

    # Class constructor
    def __init__(
        self,
        cst,
        ct,
        pln,
        projection="dose",
        solver="scipy",
        linear_solver="L-BFGS-B",
        max_iter=500,
        max_cpu_time=3000,
    ):

        # Start constructor runtime recording
        initStart = time()

        # Validate Inputs
        self.pln = validate_pln(pln)

        # Initialize cst, pln and dij
        self.ct = validate_ct(ct)
        self.cst = validate_cst(cst, ct=self.ct)

        # Setting up the plan configuration for the optimization algorithm. Only the RBE is
        # necessary. We will get it from the previously defined pln dictionary

        # if self.pln.radiation_mode == "photons":
        #     self.pln = {"RBE": 1.0}
        # elif self.pln.radiation_mode == "protons":
        #     self.pln = {"RBE": 1.1}

        # Get all objective functions
        self.objs = tuple(
            FluenceOptimizer.flatten(
                self.cst[struct]["dose_objective"]
                for struct in self.cst
                if self.cst[struct]["dose_objective"] is not None
            )
        )

        # Preprocess cst
        self.cst = self.adjustParamsFx(
            self.cst, self.dij
        )  # Parameter adjustment due to fractionation
        # self.cst = self.setOverlapPriorities(
        #    self.cst
        # )  # Priorization due to overlapping structures
        self.cst = self.resizeStructs(
            self.cst, self.ct, self.dij
        )  # Resizing structures to fit dose grid

        # Set initial weight vector
        self.wInit = self.initWeights(self.cst, self.dij, self.pln)

        # Get backprojection attribute based on "projection" argument
        PROJECTION = {"dose": DoseProjection, "constantRBE": ConstantRBEProjection}
        self.backProjection = PROJECTION.get(projection, "dose")(self.dij, self.pln)

        # Initialize optimization problem
        self.optProb = OptimizationProblem(self.backProjection, self.cst)

        # Set lower and upper variable bounds
        self.setLowerVarBounds([0] * len(self.wInit))
        self.setUpperVarBounds([inf] * len(self.wInit))

        # Set lower and upper constraint bounds
        self.setLowerConstrBounds([])
        self.setUpperConstrBounds([])

        # Get solver attribute based on "solver" argument
        SOLVER = {"scipy": SciPySolver}

        self.wInit = self.wInit.reshape(-1)
        self.solver = SOLVER.get(solver, "scipy")(
            len(self.wInit),  # number of variables
            len(self.lowerConstrBound),  # number of constraints
            self.optProb,  # Instance of the optimization problem
            self.lowerVarBound,  # lower variable bounds
            self.upperVarBound,  # upper variable bounds
            self.lowerConstrBound,  # lower constraint bounds
            self.upperConstrBound,  # upper constraint bounds
            linear_solver=linear_solver,
            max_iter=max_iter,  # maximum number of iterations
            max_cpu_time=max_cpu_time,
        )  # maximum cpu time

        # Initialize attributes for documenting results
        self.wOpt = None  # optimal fluence vector
        self.dOpt = None  # optimized dose distribution
        self.optInfo = None  # solver information about optimization process

        # End constructor runtime recording
        initEnd = time()
        self.initTime = initEnd - initStart

    # Print formatted class attributes
    def __str__(self):
        return "\n".join(
            ("FluenceOptimizer class attributes:", "------------", str((*self.__dict__,)))
        )

    # Adjust dose parameters according to selected number of fractions
    def adjustParamsFx(self, cst, dij):

        # Adjust single objective
        def adjust(obj):

            # Get dose-related parameters depending on parameter type (iterable or scalar)
            ids = tuple(item[0] for item in enumerate(obj.parameter_types) if item[1] == "dose")
            if len(ids) > 1 or (
                len(ids) == 1
                and isinstance(obj.parameters, (tuple, list))
                and len(obj.parameters) > 1
            ):
                for idx in ids:
                    obj.set_parameters(obj.get_parameters()[idx] / dij["numOfFractions"])
            else:
                obj.set_parameters(obj.get_parameters() / dij["numOfFractions"])

            # Set adjusted_params attribute to true for preventing re-adjustment
            obj.adjusted_params = True

        # Run adjust function for all objectives that have not been adjusted yet and contain
        # dose parameters
        for obj in (
            obj for obj in self.objs if not obj.adjusted_params and "dose" in obj.parameter_types
        ):
            adjust(obj)

        return cst

    # Prioritize structures according to priority in case of overlaps
    def setOverlapPriorities(self, cst):

        # Remove overlap for single structure
        def removeOverlap(structA):

            # Get list of raw indices from structures with higher priority and associated
            # optimization component
            ids = [
                cst[structB]["raw_indices"]
                for structB in cst
                if (
                    cst[structB]["parameters"]["Priority"] < cst[structA]["parameters"]["Priority"]
                )
                and (
                    cst[structB]["dose_objective"] is not None
                    or cst[structB]["doseConstraint"] is not None
                )
            ]

            # Compute union over index list
            union = reduce(union1d, ids, -1)

            # Remove shared indices to get indices after prioritization
            # and enter into cst dictionary
            cst[structA]["prior_indices"] = setdiff1d(cst[structA]["raw_indices"], union)

        # Run removeOverlap function for all structures
        ThreadPoolExecutor().map(removeOverlap, (*cst,))

        return cst

    # Interpolate structures between ct and dose grid
    def resizeStructs(self, cst: StructureSet, ct: CT, dij: Dij):
        resampled_ct = ct.resample_to_grid(dij.dose_grid)
        cst_new = cst.resample_on_new_ct(resampled_ct)
        return cst_new

    # Initialize fluence weight vector
    def initWeights(self, cst, dij, pln):

        # self.logger.dispInfo('\t Initializing fluence weight vector ...')

        # Get target structures and indices
        structs = tuple(
            struct
            for struct in cst
            if (cst[struct]["type"] == "TARGET")
            and (
                cst[struct]["dose_objective"] is not None
                or cst[struct]["doseConstraint"] is not None
            )
        )
        ids = hstack([cst[struct]["resized_indices"] for struct in structs])

        # Get dose parameter values from structures with objective function
        def getDoseParams(struct):

            # Get indices of dose parameters
            pos = tuple(
                item[0]
                for item in enumerate(cst[struct]["dose_objective"].parameter_types)
                if item[1] == "dose"
            )

            return (
                (cst[struct]["dose_objective"].parameters[p] for p in pos)
                if len(pos) > 1
                else cst[struct]["dose_objective"].parameters
            )

        # Compute maximum dose parameter value
        dmax = max(ThreadPoolExecutor().map(getDoseParams, structs))

        # Compute initial beam fluence vector
        wOnes = ones((dij["totalNumOfBixels"], 1))
        wInit = wOnes * dmax / (pln["RBE"] * (dij["physicalDose"][ids, :] @ wOnes).mean())

        # Fixes problem of wrong dimensionality
        wInit = wInit.reshape(-1)

        return wInit

    # Set lower bound to decision variables
    def setLowerVarBounds(self, val):
        self.lowerVarBound = val

    # Set upper bound to decision variables
    def setUpperVarBounds(self, val):
        self.upperVarBound = val

    # Set lower bounds to constraint functions
    def setLowerConstrBounds(self, val):
        self.lowerConstrBound = val

    # Set upper bounds to constraint functions
    def setUpperConstrBounds(self, val):
        self.upperConstrBound = val

    # Run optimization process
    def solve(self):

        # Start solver runtime recording
        solveStart = time()

        # Solve for w and compute optimized dose distribution
        self.wOpt, self.optInfo = self.solver.start(self.wInit)
        self.dOpt = self.computeDose3D(self.wOpt)

        # End solver runtime recording
        solveEnd = time()

        self.solveTime = solveEnd - solveStart

    # Compute optimized 3-dimensional dose distribution from optimal fluence vector
    def computeDose3D(self, wOpt):

        # Compute optimized dose vector
        dOpt = self.dij["physicalDose"] @ self.wOpt

        # Reshape 1-dimensional dose to 3-dimensional dose array
        dOpt = dOpt.reshape(self.dij["cubeDim"], order="F")

        # Interpolate dose array to fit ct grid
        zooms = (
            self.ct["cubeDim"][j] / self.dij["cubeDim"][j]
            for j in range(0, len(self.ct["cubeDim"]))
        )
        dOpt = zoom(dOpt, zooms, order=1) * self.pln["RBE"]

        return dOpt

    # Get optimization results (optimal fluence, dose and optimizer information)
    def getResults(self, add_info=False):
        return (
            {"wOpt": self.wOpt, "dOpt": self.dOpt, "optInfo": self.optInfo}
            if add_info
            else {"wOpt": self.wOpt, "dOpt": self.dOpt}
        )
