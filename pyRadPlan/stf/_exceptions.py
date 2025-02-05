"""Module _exceptions.py
This module contains all custom exceptions for the stf module.
"""

from pyRadPlan.core import PyRadPlanException


class GeometryError(PyRadPlanException):
    """An exception raised when there is an error in the geometry during
    generation of the stf.
    """

    def __init__(self, message: str):
        super().__init__(message)
