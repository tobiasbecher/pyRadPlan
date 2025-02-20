"""Scalarization Strategy Base Classes for Planning Problems."""

from typing import ClassVar, Callable
from abc import ABC, abstractmethod
import numpy as np

class ScalarizationStrategyBase(ABC):
    """
    Abstract Base Class for Scalarization Strategy Implementations / Interfaces.

    Attributes
    ----------
    name : ClassVar[str]
        Full name of the scalarization strategy
    short_name : ClassVar[str]
        Short name of the scalarization strategy
    """

    name = ClassVar[str]
    short_name = ClassVar[str]

    # properties
    # TODO

    def __init__(self, 
            evaluate_objectives: callable[np.ndarray[float],list[np.ndarray[float]]],
            evaluate_constraints: callable[np.ndarray[float],list[np.ndarray[float]]],
            evaluate_x_gradients,
            evaluate__jacobian,
            etc,
            parameters: dict
        ):
        # Implementations of class should also manage the concatenating the additional variables and separating them 
        pass

    def __repr__(self) -> str:
        return f"Scalarization Strategy {self.name} ({self.short_name})"
    


    @abstractmethod
    def variable_upper_bounds() -> np.ndarray[float]:
        pass

    @abstractmethod
    def variable_lower_bounds() -> np.ndarray[float]:
        pass

    @abstractmethod
    def get_linear_constraints(self) -> dict[Index, LinearConstraint]:
        pass

    @abstractmethod
    def get_nonlinear_constraints(self) -> dict[Index, NonlinearConstraint]:
        pass

    @abstractmethod
    def evaluate_objective(x: np.ndarray) -> float:
        print("This is not the same as evaluating objectives. E.g. make weighted sum")

    @abstractmethod
    def evaluate_constraints(x: np.ndarray) -> np.ndarray:
        print("Most of the time this will be the objective constraints and the constraints from the scalarization method")

    def solve(self, x: np.ndarray[float]) -> np.ndarray[float]:
        print("Do something")
        self._call_solver_interface(self.solver, self.solver_params)

    def is_objective_convex() -> bool:
        pass

    def _call_solver_interface(solver: str, params: dict) -> np.ndarray[float]:
        pass