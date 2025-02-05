"""Helper Functions relating to conversion between numpy and SimpleITK data
types.
"""

from typing import Literal
import SimpleITK as sitk
import numpy as np
from ._grids import Grid


def sitk_mask_to_linear_indices(mask: sitk.Image, order="sitk") -> np.ndarray:
    """
    Convert a SimpleITK mask to linear indices.

    Parameters
    ----------
    mask : sitk.Image
        The SimpleITK image mask to be converted.
    order : str, optional
        The ordering of the indices. Can be 'sitk' for Fortran-like index
        ordering or 'numpy' for C-like index ordering. Default is 'sitk'.

    Returns
    -------
    np.ndarray
        A 1D numpy array of linear indices where the mask is non-zero.

    Raises
    ------
    ValueError
        If the ordering is not 'sitk' or 'numpy'.
    """

    arr = sitk.GetArrayViewFromImage(mask)
    if order == "sitk":
        return np.nonzero(arr.ravel(order="F"))[0]
    if order == "numpy":
        return np.flatnonzero(arr)
    raise ValueError("Invalid ordering. Must be 'sitk' or 'numpy'.")


def linear_indices_to_sitk_mask(
    indices: np.ndarray, ref_image: sitk.Image, order="sitk"
) -> sitk.Image:
    """
    Convert linear indices to a SimpleITK mask.

    Parameters
    ----------
    indices : np.ndarray
        A 1D numpy array of linear indices where the mask is non-zero.
    ref_image : sitk.Image
        The reference image on which the mask is defined.
    order : str, optional
        The ordering of the indices. Can be 'sitk' for Fortran-like index
        ordering or 'numpy' for C-like index ordering. Default is 'sitk'.

    Returns
    -------
    sitk.Image
        A SimpleITK image mask with the indices set to non-zero.

    Raises
    ------
    ValueError
        If the ordering is not 'sitk' or 'numpy'.
    """

    arr = np.zeros_like(sitk.GetArrayViewFromImage(ref_image), dtype=np.uint8)

    if order == "sitk":
        arr.T.flat[indices] = 1
    elif order == "numpy":
        arr.flat[indices] = 1
    else:
        raise ValueError("Invalid ordering. Must be 'sitk' or 'numpy'.")

    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(ref_image)

    return mask


def linear_indices_to_grid_coordinates(
    indices: np.ndarray, grid: Grid, index_type: Literal["numpy", "sitk"] = "numpy"
) -> np.ndarray:
    """
    Convert linear indices to gridcoordinates.

    Parameters
    ----------
    indices : np.ndarray
        A 1D numpy array of linear indices where the mask is non-zero.
    grid : Grid
        The image grid on which the indices lie.
    index_type : Literal["numpy", "sitk"], optional
        The ordering of the indices. Can be 'sitk' for Fortran-like index
        ordering or 'numpy' for C-like index ordering. Default is 'numpy'.

    Returns
    -------
    np.ndarray
        A 2D numpy array of image coordinates.
    """

    if index_type == "numpy":
        z, y, x = np.unravel_index(indices, grid.dimensions[::-1])
    elif index_type == "sitk":
        x, y, z = np.unravel_index(indices, grid.dimensions)
    else:
        raise ValueError("Invalid index type. Must be 'numpy' or 'sitk'.")

    v = np.array([x, y, z])

    spacing_diag = np.diag([grid.resolution["x"], grid.resolution["y"], grid.resolution["z"]])
    origin = grid.origin

    physical_point = origin + np.matmul(np.matmul(grid.direction, spacing_diag), v).T

    return physical_point


def linear_indices_to_image_coordinates(
    indices: np.ndarray, image: sitk.Image, index_type: Literal["numpy", "sitk"] = "numpy"
) -> np.ndarray:
    """
    Convert linear indices to image coordinates.

    Parameters
    ----------
    indices : np.ndarray
        A 1D numpy array of linear indices where the mask is non-zero.
    image : sitk.Image
        The reference image on which the mask is defined.
    index_type : Literal["numpy", "sitk"], optional
        The ordering of the indices. Can be 'sitk' for Fortran-like index
        ordering or 'numpy' for C-like index ordering. Default is 'numpy'.

    Returns
    -------
    np.ndarray
        A 2D numpy array of image coordinates.
    """

    grid = Grid.from_sitk_image(image)
    return linear_indices_to_grid_coordinates(indices, grid, index_type)
