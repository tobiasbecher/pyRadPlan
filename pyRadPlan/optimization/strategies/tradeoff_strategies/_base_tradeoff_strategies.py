from abc import ABC
from typing import ClassVar
import numpy as np

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
    ScalarizationMethod: tbd

    def __init__(self, params: dict):
        pass

    def solve(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        pass