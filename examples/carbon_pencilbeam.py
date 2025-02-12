import logging

try:
    from importlib import resources  # Standard from Python 3.9+
except ImportError:
    import importlib_resources as resources  # Backport for older versions

import numpy as np

from pyRadPlan import (
    IonPlan,
    load_patient,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
)

logging.basicConfig(level=logging.INFO)

#  Read patient from provided TG119.mat file and validate data
path = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
ct, cst = load_patient(path)

# Create a plan object
pln = IonPlan(radiation_mode="carbon", machine="Generic")
pln.prop_stf = {"bixel_width": 4}
pln.prop_dose_calc = {"calc_bio_dose": True}
pln.prop_opt = {"solver": "scipy"}

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
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["effect"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="dimensionless",
)
