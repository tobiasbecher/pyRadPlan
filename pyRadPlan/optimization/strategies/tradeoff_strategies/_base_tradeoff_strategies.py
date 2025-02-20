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
    ScalarizationStrategy: str

    def __init__(self):
        pass

    def solve(x: np.ndarray[float]) -> list[np.ndarray[float]]:
        pass