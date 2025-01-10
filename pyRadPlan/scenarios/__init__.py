"""Sceanrio Models for uncertainty quantification, robust optimization and
robustness analysis.

This module provides functionality for creating and validating scenario models.
"""

from typing import Union
from warnings import warn
from pydantic.alias_generators import to_snake
from pyRadPlan.ct import CT, validate_ct
from ._base import ScenarioModel
from ._nominal import NominalScenario


def available_scenario_models() -> list[str]:
    """Returns a list of available scenario models."""

    return ["nomScen"]


def create_scenario_model(
    model_def: Union[ScenarioModel, Union[str, dict]] = "nomScen", ct: Union[CT, dict] = None
) -> ScenarioModel:
    """
    Returns a scenario model object.

    Parameters
    ----------
    model_name : str
        The name of the scenario model.

    Returns
    -------
    ScenarioModel
        The scenario model object.
    """

    if isinstance(ct, dict):
        ct = validate_ct(ct)

    if isinstance(model_def, ScenarioModel):
        return model_def

    if isinstance(model_def, dict):
        model_name = model_def.get("model", None)
        if model_name is None:
            raise ValueError("Scenario model name not provided in the model definition")

        model = create_scenario_model(model_name, ct)
        for key in model_def:
            if key == "model":
                continue
            attr_name = key
            if not hasattr(model, attr_name):
                attr_name = to_snake(key)
                if not hasattr(model, attr_name):
                    warn(f"Unknown attribute {key} in scenario model definition")
                    continue

            setattr(model, attr_name, model_def[key])

        return model

    if isinstance(model_def, str):
        model_name = model_def

        if model_name == "nomScen":
            return NominalScenario(ct)
        if model_name == "wcScen":
            raise NotImplementedError("Worst Case Scenarios are not implemented yet")
        if model_name == "impScen":
            raise NotImplementedError(
                "Gridded Importance Weighted Scenarios are not implemented yet"
            )
        if model_name == "rndScen":
            raise NotImplementedError("Random Scenarios are not implemented yet")
        raise ValueError(f"Unknown scenario model: {model_name}")

    raise ValueError("Invalid scenario model definition")


def validate_scenario_model(
    model_def: Union[ScenarioModel, Union[str, dict]], ct: Union[CT, dict] = None
) -> ScenarioModel:
    """
    Validates a scenario model input and returns a scenario model object.

    Parameters
    ----------
    model_def : Union[ScenarioModel, Union[str, dict]]
        The scenario model object.

    Returns
    -------
    ScenarioModel
        The scenario model object.
    """

    return create_scenario_model(model_def, ct)


__all__ = ["ScenarioModel", "NominalScenario", "create_scenario_model"]
