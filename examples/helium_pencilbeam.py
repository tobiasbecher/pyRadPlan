import logging

try:
    from importlib import resources  # Standard from Python 3.9+
except ImportError:
    import importlib_resources as resources  # Backport for older versions
import numpy as np
import SimpleITK as sitk
import pymatreader
import matplotlib.pyplot as plt

from pyRadPlan import (
    IonPlan,
    validate_ct,
    validate_cst,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
)

logging.basicConfig(level=logging.INFO)

#  Read patient from provided TG119.mat file and validate data
path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
tmp = pymatreader.read_mat(path)
ct = validate_ct(tmp["ct"])
cst = validate_cst(tmp["cst"], ct=ct)

# Create a plan object
pln = IonPlan(radiation_mode="helium", machine="Generic")
pln.prop_stf = {"bixel_width": 4}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Optimize
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Result
result = dij.compute_result_ct_grid(fluence)

# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# Visualize
cube_hu = sitk.GetArrayViewFromImage(ct.cube_hu)
plt.imshow(cube_hu[view_slice, :, :], cmap="gray")

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

# Now let's visualize the VOIs from the StructureSet.
for v, voi in enumerate(cst.vois):
    mask = sitk.GetArrayViewFromImage(voi.mask)
    color = plt.cm.cool(v / len(cst.vois))  # Select color based on colormap 'cool'
    plt.contour(
        mask[view_slice, :, :],
        levels=[0.5],
        colors=[color],
        linewidths=1,
    )
dose_array = sitk.GetArrayViewFromImage(result["physical_dose"])
plt.imshow(
    dose_array[view_slice, :, :],
    cmap="jet",
    interpolation="nearest",
    alpha=0.5 * (dose_array[view_slice, :, :] > 0.02),
)
plt.colorbar(label="dose [Gy]")
plt.title(f"Helium Dose (Slice z={view_slice})")
plt.show()
plt.savefig(f"{pln.radiation_mode}_physical_dose.png")
plt.clf()
