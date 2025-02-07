from typing import Optional, Union, Literal

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pint

ureg = pint.UnitRegistry()

from pyRadPlan import CT, validate_ct, StructureSet, validate_cst


def plot_slice(
    ct: Optional[Union[CT, dict]] = None,
    cst: Optional[Union[StructureSet, dict, list]] = None,
    overlay: Optional[Union[sitk.Image, np.ndarray]] = None,
    view_slice: Optional[int] = None,
    plane: Union[Literal["axial", "coronal", "sagittal"], int] = "axial",
    overlay_alpha: float = 0.5,
    overlay_unit: Union[str, pint.Unit] = pint.Unit(""),
    overlay_rel_threshold: float = 0.01,
    contour_line_width: float = 1.0,
    title: Optional[str] = None,
):
    """Plot a slice of the CT with overlay.

    Parameters
    ----------
    ct : CT
        The CT object.
    cst : StructureSet
        The StructureSet object.
    """

    if ct is not None:
        ct = validate_ct(ct)
        cube_hu = sitk.GetArrayViewFromImage(ct.cube_hu)

    if cst is not None:
        cst = validate_cst(cst)

    if ct is None and cst is None:
        raise ValueError("Nothing to visualize!")

    if isinstance(plane, str):
        if plane == "axial":
            plane = 0
        elif plane == "coronal":
            plane = 1
        elif plane == "sagittal":
            plane = 2

    if isinstance(overlay_unit, str):
        overlay_unit = ureg(overlay_unit)

    slice_indexing = tuple(slice(None) if i != plane else view_slice for i in range(3))

    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False,
    )

    # Visualize the CT slice
    if ct is not None:
        plt.imshow(cube_hu[slice_indexing], cmap="gray")

    # Now let's visualize the VOIs from the StructureSet.
    if cst is not None:
        for v, voi in enumerate(cst.vois):
            mask = sitk.GetArrayViewFromImage(voi.mask)
            cmap = plt.colormaps["cool"]
            color = cmap(v / len(cst.vois))  # Select color based on colormap 'cool'
            plt.contour(
                mask[view_slice, :, :],
                levels=[0.5],
                colors=[color],
                linewidths=1,
            )

    if overlay is not None:
        if isinstance(overlay, sitk.Image):
            overlay = sitk.GetArrayViewFromImage(overlay)

        if not isinstance(overlay, np.ndarray):
            raise ValueError("Overlay must be a numpy array or SimpleITK image.")

        plt.imshow(
            overlay[slice_indexing],
            cmap="jet",
            interpolation="nearest",
            alpha=overlay_alpha
            * (overlay[slice_indexing] > overlay_rel_threshold * np.max(overlay)),
        )
        plt.colorbar(label=format(overlay_unit, "~P"))

    plt.title(f"Slice z={view_slice}")
    plt.show()
