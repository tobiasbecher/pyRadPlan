"""Base Implementation for objective functions."""
from abc import abstractmethod
from typing import ClassVar, get_type_hints, Any

from pydantic import computed_field, Field

from pyRadPlan.core.datamodel import PyRadPlanBaseModel

#%% Class definition


class Objective(PyRadPlanBaseModel):
    """
    Base class for objective functions in the optimization problem.

    Attributes
    ----------
    name : str
        Name of the objective function.
    parameter_names : list of str
        Names of the parameters of the objective function.
    parameter_types : list of str
        Types of the parameters of the objective function.
    parameters : list of float
        Parameters of the objective function.
    weight : float
        Weight of the objective function.
    """

    name: ClassVar[str]
    parameter_names: ClassVar[list[str]]
    has_hessian: ClassVar[bool] = False
    priority: float = Field(default=1.0, ge=0.0)

    @abstractmethod
    def compute_objective(self, *args):
        """Computes the objective function."""

    @abstractmethod
    def compute_gradient(self, *args):
        """Computes the objective gradient."""

    def compute_hessian(self, *args):
        """Computes the objective Hessian."""
        return None

    @computed_field
    @property
    def parameter_types(self) -> list[Any]:
        """Parameter Types."""
        types = get_type_hints(self.__class__)
        field_types = []
        for p_name in self.parameter_names:
            p_fields = self.model_fields[p_name]
            field_type = types[p_fields]
            field_types.append(field_type)

        return field_types

    @computed_field
    @property
    def parameters(self) -> list[Any]:
        """Parameter values."""
        return [getattr(self, p_name) for p_name in self.parameter_names]
