from abc import ABC, abstractmethod
from typing import Union, ClassVar
import warnings
import logging

import numpy as np

from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.dij import Dij, validate_dij

logger = logging.getLogger(__name__)


class OptimizationProblem(ABC):
    """Abstract Class for all Treatment Planning Optimization Problems."""

    # Constant, Abstract properties are realized as ClassVars
    short_name: ClassVar[str]
    name: ClassVar[str]

    # Properties
    optimizer: Union[str, dict]
    apply_overlap: bool

    def __init__(self, pln: Union[Plan, dict] = None):

        self._ct = None
        self._cst = None
        self._stf = None
        self._dij = None

        self.optimizer = "scipy"

        if pln is not None:
            pln = validate_pln(pln)
            self.assign_properties_from_pln(pln)

        self.apply_overlap = True

    def solve(
        self,
        ct: Union[CT, dict],
        cst: Union[StructureSet, dict],
        stf: Union[SteeringInformation, dict],
        dij: Union[Dij, dict],
    ) -> dict:
        """Solve the optimization problem."""
        self._ct = validate_ct(ct)
        self._cst = validate_cst(cst)
        self._stf = validate_stf(stf)
        self._dij = validate_dij(dij)

        self._initialize()
        self._optimize()
        return self._finalize()

    @abstractmethod
    def _objective_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the objective functions."""
        pass

    @abstractmethod
    def _objective_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective jacobian."""
        return None

    def _objective_hessian(self, x: np.ndarray) -> np.ndarray:
        """Define the objective hessian."""
        return {}

    def _constraint_functions(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint functions."""
        return None

    def _constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Define the constraint jacobian."""
        return None

    def _constraint_jacobian_structure(self) -> np.ndarray:
        """Define the constraint jacobian structure."""
        return None

    def _variable_bounds(self, x: np.ndarray) -> np.ndarray:
        """Define the variable bounds."""
        return {}

    def _initialize(self):
        """Initialize the optimization problem."""

        # resampling to dose-grid
        self._ct = self._ct.resample_to_grid(self._dij.dose_grid)

        # apply overlap priorities
        if self.apply_overlap:
            self._cst = self._cst.apply_overlap_priorities().resample_on_new_ct(self._ct)
        else:
            self._cst = self._cst.resample_on_new_ct(self._ct)

        # sanitize objectives and constraints

        # set solver options

        # initial point

    @abstractmethod
    def _optimize(self):
        """Optimize the optimization problem."""

    @abstractmethod
    def _finalize(self) -> np.ndarray:
        """Finalize the optimization problem."""

    def assign_properties_from_pln(self, pln: Plan, warn_when_property_changed: bool = False):
        """
        Assign properties from a Plan object to the Dose Engine.
        This includes the Scenario Model and the Biological Model, and any
        other properties that
        can be stored in the prop_opt dictionary within the Plan object.
        This function will check if a property exists for the dose opti_prob
        and,
        if yes, set it.

        Parameters
        ----------
        pln : Plan
            The Plan object to assign properties from.
        warn_when_property_changed : bool
            Whether to warn when properties are changed.
        """

        # Set Scenario Model
        if hasattr(pln, "mult_scen"):
            self.mult_scen = pln.mult_scen

        # Assign Biologival Model
        if hasattr(pln, "bio_param"):
            self.bio_param = pln.bio_param  # TODO: No bio_param yet

        if not isinstance(warn_when_property_changed, bool):
            warn_when_property_changed = False

        # Overwrite default properties within the opti_prob
        # with the ones given in the prop_opt dict
        if hasattr(pln, "prop_opt") and isinstance(
            pln.prop_opt, dict
        ):  # TODO: This is not tested yet
            prop_dict = pln.prop_opt
            if (
                "opti_prob" in prop_dict
                and prop_dict["opti_prob"]
                and prop_dict["opti_prob"] != self.short_name
            ):
                raise ValueError(
                    f"Inconsistent dose opti_probs given! pln asks for '{prop_dict['opti_prob']}', "
                    f"but you are using '{self.short_name}'!"
                )
            prop_dict.pop("opti_prob", None)
        else:
            prop_dict = {}

        fields = prop_dict.keys()

        # Set up warning message
        if warn_when_property_changed:
            warning_msg = "Property in Dose Engine overwritten from pln.prop_opt"
        else:
            warning_msg = None

        for field in fields:
            if not hasattr(self, field):
                warnings.warn('Property "{}" not found in Dose Engine!'.format(field))
            else:
                if warn_when_property_changed and warning_msg:
                    logger.warning(warning_msg + f": {field}")

            setattr(self, field, prop_dict[field])
