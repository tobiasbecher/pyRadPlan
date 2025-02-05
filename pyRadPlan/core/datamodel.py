"""Module datamodel.py
Basic Model for all pyRadPlan Datastructures.
"""

from typing import Any, Union
from pydantic import (
    AliasGenerator,
    BaseModel,
    ConfigDict,
)
from pydantic.alias_generators import to_camel
from copy import deepcopy


class PyRadPlanBaseModel(BaseModel):
    """
    Base class for all pyRadPlan data structures, especially the ones that
    should be matRad
    compatible
    This class extends Pydantic's BaseModel.

    Attributes
    ----------
    model_config : ConfigDict
        Configuration for the model, including alias generation, population by name, arbitrary
        types allowed, assignment validation, and attribute creation from dictionary.
    """

    model_config = ConfigDict(
        alias_generator=AliasGenerator(alias=to_camel),
        populate_by_name=True,  # Allows both snake_case and camelCase attributes
        arbitrary_types_allowed=True,  # Allows arbitrary types in the model (will be casted)
        validate_assignment=True,  # Validate assignment of values to fields
        # (not only during construction)
        from_attributes=True,  # Allows to create a model from a dictionary
    )

    def to_matrad(self, context: Union[str, dict] = "mat-file") -> Any:
        """
        Interface method to serialize a pyradplan datastructure to be matRad
        compatible.

        Parameters
        ----------
        context : str, optional
            The context in which the datastructure should be serialized, by default 'mat-file'.

        Returns
        -------
        Any
            A datastructre compatible with matRad in the given context

        Notes
        -----
        Currently, the only supported context is 'mat-file'. In the future, this could be
        extended to support other contexts, such as direct calling via the matlab engine or
        oct2py.
        """
        self_copy = deepcopy(self)
        if isinstance(context, dict):
            if "matRad" not in context:
                context.update({"matRad": "mat-file"})
        else:
            context = {"matRad": context}

        if context["matRad"] != "mat-file":
            raise ValueError(f"Context {context} not supported")

        # Standard is a model_dump using above alias and context information
        return self_copy.model_dump(by_alias=True, context=context)
