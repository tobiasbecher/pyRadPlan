from abc import ABC
from typing import ClassVar
import numpy as np
from ..scalarization_strategies._base_scalarization_strategies import ScalarizationStrategyBase

class TradeoffStrategyBase(ABC):
    """
    To be written later
    Abstract class for tradeoff exploration methods

    Parameters
    -----------

    Attributes
    ----------
    """

    short_name: ClassVar[str]
    name: ClassVar[str]
    scalarization_strategy: str

    def __init__(self,callbacks: dict[str, callable],scalarization_strategy: ScalarizationStrategyBase):
        self.callbacks = callbacks
        self.scalarization_strategy = scalarization_strategy

    def solve(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        pass