"""Placeholder for future implementation."""

from pydantic import Field
from typing import ClassVar, List
from ._base import ParticleAccelerator


class VHEEAccelerator(ParticleAccelerator):
    """Machine Model for Very High Energy Electron (VHEE) Accelerators."""

    name: str = Field(default="VHEE", alias="machine")
    key: str = "vhee"
    radiation_mode: str = Field(
        default="electrons", pattern="^(electrons)$", validate_default=True
    )
    _possible_radiation_modes: ClassVar[List[str]] = ["electrons"]
    # energies and sad fields are inherited from ParticleAccelerator and must be provided when instantiating
    pass
