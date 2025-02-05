"""Module: _base.py
This module contains the base class for scenario models.

Classes
-------
- ScenarioModel: Abstract base class for scenario models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import numpy as np
from copy import deepcopy
from pyRadPlan.ct import validate_ct


class ScenarioModel(ABC):
    """Abstract base class for scenario models.

    Attributes
    ----------
    range_rel_sd : float
        The relative standard deviation for range.
    range_abs_sd : float
        The absolute standard deviation for range.
    shift_sd : Tuple[float]
        The standard deviation for shift in x, y, and z directions.
    wc_sigma : float
        The sigma value for the wc_factor.
    ct_scen_prob : List[Tuple[int,float]]
        The probability of each CT scenario.

    Methods
    -------
        update_scenarios() -> np.ndarray[float]
            Abstract method to update scenarios.
        extract_single_scenario(scen_num: int) -> ScenarioModel
            Extracts a single scenario based on its number.
        list_all_scenarios()
            Lists all scenarios.
        sub2scenIx(ctScen: int, shift_scen: int, range_shift_scen: int) -> int
            Converts CT scenario, shift scenario, and range shift scenario to scenario index.
        scenNum(fullScenIx: int) -> int
            Converts full scenario index to scenario number.
    """

    # Constructor
    def __init__(self, ct=None):
        if ct is not None:
            ct = validate_ct(ct)
            self._num_of_ct_scen = ct.num_of_ct_scen
            self._num_of_available_ct_scen = ct.num_of_ct_scen
        else:
            self._num_of_available_ct_scen = 1
            self._num_of_ct_scen = 1

        self._ct_scen_prob = [
            (i, 1 / self._num_of_ct_scen) for i in range(0, self._num_of_ct_scen)
        ]

        self.update_scenarios()

    # Abstract property override
    @property
    def name(self) -> str:
        """Name of the scenario model."""
        return self._name

    @name.setter
    def name(self, value: str):
        raise AttributeError("Cannot set value directly. Use subclass implementation.")

    @property
    def short_name(self) -> str:
        """Short name of the scenario model."""
        return self._short_name

    @short_name.setter
    def short_name(self, value: str):
        raise AttributeError("Cannot set value directly. Use subclass implementation.")

    # Main Uncertainty Model Properties
    @property
    def range_rel_sd(self) -> float:
        """The relative standard deviation for range."""
        return self._range_rel_sd

    @range_rel_sd.setter
    def range_rel_sd(self, value: float):
        self._range_rel_sd = value
        self.update_scenarios()

    @property
    def range_abs_sd(self) -> float:
        """The absolute standard deviation for range."""
        return self._range_abs_sd

    @range_abs_sd.setter
    def range_abs_sd(self, value: float):
        self._range_abs_sd = value
        self.update_scenarios()

    @property
    def shift_sd(self) -> Tuple[float]:
        """The standard deviation for shift in x, y, and z directions."""
        return self._shift_sd

    @shift_sd.setter
    def shift_sd(self, value: Tuple[float]):
        self._shift_sd = value
        self.update_scenarios()

    @property
    def wc_sigma(self) -> float:
        """The sigma value for the wc_factor."""
        return self._wc_sigma

    @wc_sigma.setter
    def wc_sigma(self, value: float):
        self._wc_sigma = value
        self.update_scenarios()

    @property
    def ct_scen_prob(self) -> List[Tuple[int, float]]:
        """The probability of each CT scenario."""
        return self._ct_scen_prob

    @ct_scen_prob.setter
    def ct_scen_prob(self, value: List[Tuple[int, float]]):
        self._ct_scen_prob = value
        self.update_scenarios()

    # Dependent properties
    @property
    def wc_factor(self) -> float:
        """Wc_factor is just an alias for wc_sigma."""
        return self.wc_sigma

    @wc_factor.setter
    def wc_factor(self, value: float):
        self.wc_sigma = value

    # Name Properties
    _name: str
    _short_name: str

    # Main uncertainty model properties
    _range_rel_sd: float = 3.5
    _range_abs_sd: float = 1
    _shift_sd: Tuple[float] = (2.25, 2.25, 2.25)
    _wc_sigma: float = 1
    _ct_scen_prob: List[Tuple[int, float]] = [(0, 1.0)]

    # Other properties with protected access
    _num_of_ct_scen: int
    _num_of_available_ct_scen: int
    _ct_scen_ix: np.ndarray[int]
    _iso_shift: np.ndarray[float]
    _rel_range_shift: np.ndarray[float]
    _abs_range_shift: np.ndarray[float]
    _tot_num_shift_scen: int
    _tot_num_range_scen: int
    _tot_num_scen: int
    _scen_for_prob: np.ndarray[float]
    _scen_prob: np.ndarray[float]
    _scen_weight: np.ndarray[float]
    _scen_mask: np.ndarray[bool]
    _linear_mask: np.ndarray[int]

    @property
    def num_of_ct_scen(self) -> Optional[int]:
        """Number of CT scenarios in the model."""
        return self._num_of_ct_scen

    @property
    def num_of_available_ct_scen(self) -> int:
        """Number of totally available CT scenarios in the CT."""
        return self._num_of_available_ct_scen

    @property
    def ct_scen_ix(self) -> np.ndarray[int]:
        """CT scenario indices of the used CT scenarios."""
        return self._ct_scen_ix

    @property
    def iso_shift(self) -> np.ndarray[float]:
        """Isocenter shift values."""
        return self._iso_shift

    @property
    def rel_range_shift(self) -> float:
        """Relative range shift value."""
        return self._rel_range_shift

    @property
    def abs_range_shift(self) -> float:
        """Absolute range shift value."""
        return self._abs_range_shift

    @property
    def max_abs_range_shift(self) -> float:
        """Maximum absolute range shift value."""
        return np.max(self.abs_range_shift)

    @property
    def max_rel_range_shift(self) -> float:
        """Maximum relative range shift value."""
        return np.max(self.abs_range_shift)

    @property
    def tot_num_shift_scen(self) -> int:
        """Total number of shift scenarios."""
        return self._tot_num_shift_scen

    @property
    def tot_num_range_scen(self) -> int:
        """Total number of range shift scenarios."""
        return self._tot_num_range_scen

    @property
    def tot_num_scen(self) -> Optional[int]:
        """Total number of scenarios."""
        return self._tot_num_scen

    @tot_num_scen.setter
    def tot_num_scen(self, value: Optional[int]):
        self._tot_num_scen = value

    @property
    def scen_for_prob(self) -> np.ndarray[float]:
        """Scenarios organized for probability calculation."""
        return self._scen_for_prob

    @property
    def scen_prob(self) -> np.ndarray[float]:
        """Scenarios probability matrix."""
        return self._scen_prob

    @property
    def scen_weight(self) -> np.ndarray[float]:
        """Scenarios weight matrix."""
        return self._scen_weight

    @property
    def scen_mask(self) -> np.ndarray[bool]:
        """Scenarios mask matrix describing how scenarios should be stored."""
        return self._scen_mask

    @property
    def linear_mask(self) -> np.ndarray[int]:
        """Linear mask matrix for scenario selection."""
        return self._linear_mask

    # Abstract methods
    @abstractmethod
    def update_scenarios(self) -> np.ndarray[float]:
        """This function needs to be implemented by subclasses to update
        scenarios.
        """

    @abstractmethod
    def extract_single_scenario(self, scen_num: int) -> "ScenarioModel":
        """Extract a single scenario based on its number."""

    def list_all_scenarios(self):
        """
        Print a list of all scenarios.

        This method prints the details of all scenarios in a formatted table.
        The table includes the following columns:
        - xShift: The x-axis shift value
        - yShift: The y-axis shift value
        - zShift: The z-axis shift value
        - absRng: The absolute range value
        - relRng: The relative range value
        - prob: The probability value

        Example output:
        #   xShift   yShift   zShift   absRng   relRng   prob.
        1   0.000    0.000    0.000    0.000    0.000    0.000
        2   0.100    0.200    0.300    0.400    0.500    0.600
        3   0.700    0.800    0.900    1.000    1.100    1.200
        """

        print("Listing all scenarios...")
        print("#\tctScen\txShift\tyShift\tzShift\tabsRng\trelRng\tprob.")
        for s, row in enumerate(self.scen_for_prob):
            row_str = "\t".join([f"{int(row[0]):5d}"] + [f"{x:.3f}" for x in row[1:]])
            print(f"{s + 1}\t{row_str}\t{self.scen_prob[s]:.3f}")

    def sub2scen_ix(self, ct_scen: int, shift_scen: int, range_shift_scen: int) -> int:
        """
        Convert a sub-scenario index to a scenario index.

        Parameters
        ----------
            ctScen (int): The sub-scenario index.
            shift_scen (int): The shift scenario value.
            range_shift_scen (int): The range shift scenario value.

        Returns
        -------
            int: The corresponding scenario index.

        Raises
        ------
            None

        Examples
        --------
            # Example usage
            sub2scenIx(1, 2, 3)  # Returns: 1
        """

        if self._scen_mask.shape[0] == 1:
            return self._ct_scen_ix[ct_scen]
        linear_index = np.ravel_multi_index(
            (ct_scen, shift_scen, range_shift_scen), self._scen_mask.shape
        )
        return linear_index

    def scen_num(self, full_scen_ix: int) -> int:
        """
        Returns the index of the first occurrence of `full_scen_ix` in
        `self.scen_mask`.

        Parameters
        ----------
        full_scen_ix : int
            The value to search for in `self.scen_mask`.

        Returns
        -------
        int
            The index of the first occurrence of `full_scen_ix` in `self.scen_mask`.
        """

        scen_indices = np.where(self._scen_mask.flatten())[0]
        return int(np.where(scen_indices == full_scen_ix)[0][0])

    def to_dict(self) -> Dict:
        """
        Returns a dictionary with the scenario model properties.

        Returns
        -------
        Dict
            The scenario model properties.
        """
        return {
            "model": self.short_name,
            "range_rel_sd": self.range_rel_sd,
            "range_abs_sd": self.range_abs_sd,
            "shift_sd": self.shift_sd,
            "wc_sigma": self.wc_sigma,
            "ct_scen_prob": self.ct_scen_prob,
        }

    def to_matrad(self, context: str = "mat-file") -> Dict:
        """
        Returns a dictionary with the scenario model properties formatted
        for matRad.

        Returns
        -------
        Dict
            The scenario model properties for MatRad.
        """
        self = deepcopy(self)
        if context != "mat-file":
            raise ValueError(f"Context {context} not supported")

        return {
            "model": self.short_name,
            "rangeRelSD": self.range_rel_sd,
            "rangeAbsSD": self.range_abs_sd,
            "shiftSD": self.shift_sd,
            "wcSigma": self.wc_sigma,
            # adding (1.0, 0.0) to ctScenProb to match matRad indexing
            "ctScenProb": [(a + 1, b) for (a, b) in self.ct_scen_prob],
        }
