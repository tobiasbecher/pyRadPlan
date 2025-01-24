"""Base Implementation for objective functions."""
from abc import abstractmethod
from typing import ClassVar, get_type_hints, Any, Literal, Union, Optional

from pydantic import computed_field, Field

from pyRadPlan.core.datamodel import PyRadPlanBaseModel

ParameterType = Union[Literal["reference", "numeric", "relative_volume"], list[str]]

#%% Class definition
class ParameterMetadata:
    """Parameter Metadata."""

    configurable: bool
    kind: Optional[ParameterType]

    """Configurable Parameter."""

    def __init__(self, configurable: bool = True, kind: Optional[ParameterType] = "numeric"):
        self.configurable = configurable
        self.kind = kind

    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"


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
    has_hessian: ClassVar[bool] = False
    priority: float = Field(default=1.0, ge=0.0)

    @abstractmethod
    def compute_objective(self, values):
        """Computes the objective function."""

    @abstractmethod
    def compute_gradient(self, values):
        """Computes the objective gradient."""

    def compute_hessian(self, values):
        """Computes the objective Hessian."""
        return None

    @computed_field
    @property
    def parameter_names(self) -> list[str]:
        """Parameter names."""
        return [
            name
            for name, info in self.model_fields.items()
            if any(isinstance(meta, ParameterMetadata) for meta in info.metadata)
        ]

    @computed_field
    @property
    def parameters(self) -> list[Any]:
        """Parameter values."""
        return [getattr(self, name) for name in self.parameter_names]

    @computed_field
    @property
    def parameter_types(self) -> list[ParameterType]:
        """Parameter types."""
        # Find the ParameterMetadata instance in the metadata list
        return [
            meta.kind
            for name in self.parameter_names
            for meta in self.model_fields[name].metadata
            if isinstance(meta, ParameterMetadata)
        ]
