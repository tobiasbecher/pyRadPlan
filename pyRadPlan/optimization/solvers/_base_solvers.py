from typing import ClassVar, Callable
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class SolverBase(ABC):
    name = ClassVar[str]
    short_name = ClassVar[str]

    # properties
    max_time: float
    bounds: ArrayLike

    def __init__(self):
        self.max_time = 3600
        self.bounds = [0.0, np.inf]

    @abstractmethod
    def solve(self, x0: ArrayLike) -> tuple[np.ndarray, dict]:
        pass


def __repr__(self) -> str:
    """
    The representation of Solvers.

    Returns
    -------
    string
        Class Representation as string
    """
    return f"Solver {self.name} ({self.short_name})"


class NonLinearOptimizer(SolverBase):
    max_iter: int
    abs_obj_tol: float

    objective: Callable
    gradient: Callable
    hessian: Callable
    constraints: Callable
    constraints_jac: Callable

    def __init__(self):
        super().__init__()
        self.max_iter = 500
        self.abs_obj_tol = 1e-6

        self.objective = None
        self.gradient = None
        self.hessian = None
        self.constraints = None
        self.constraints_jac = None

    def iter_func(self):
        pass
