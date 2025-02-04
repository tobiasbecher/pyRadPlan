from ._base import FluenceDependentQuantity, RTQuantity
from ._dose import Dose

QUANTITIES = {Dose.identifier: Dose}


def get_available_quantities() -> dict[str, RTQuantity]:
    """
    Obtains the available quantities in planning.

    Returns
    -------
    dict
        Dictionary with the available quantities
    """
    return QUANTITIES


def get_quantity(identifier: str) -> RTQuantity:
    """
    Obtains the quantity from name.

    Parameters
    ----------
    identifier : str
        The identifier of the quantity

    Returns
    -------
    RTQuantity
        The quantity
    """
    return QUANTITIES[identifier]


__all__ = [
    "FluenceDependentQuantity",
    "Dose",
    "RTQuantity",
    "get_available_quantities",
    "get_quantity",
]
