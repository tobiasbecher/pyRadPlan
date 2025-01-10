"""Running the photon pencil-beam dose calculation engine."""

# Standard Library Imports
from importlib import resources
import logging

# Third Party Imports from pyRadPlan requirements
import pymatreader
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# pyRadPlan Module Imports
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.plan import PhotonPlan
from pyRadPlan.stf import StfGeneratorPhotonIMRT
from pyRadPlan.dose import calc_dose_influence

# Configure the Logger to show you debug information
logging.basicConfig(level=logging.INFO)
logging.getLogger("pyRadPlan").setLevel(logging.DEBUG)

# We use importlib resources to load the TG119.mat file from the pyRadPlan.data.phantoms package
files = resources.files("pyRadPlan.data.phantoms")
path = files.joinpath("TG119.mat")
tmp = pymatreader.read_mat(path)

# The validation functions for ct and cst allow us to create valid CTs and StructureSets
ct = validate_ct(tmp["ct"])
cst = validate_cst(tmp["cst"], ct=ct)

# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# We store images as sitk.Image objects, but we can easily convert them to numpy arrays for
# matplotlib. Taking a "view" here means we do not perform a deep copy of the image data.
cube_hu = sitk.GetArrayViewFromImage(ct.cube_hu)
plt.imshow(cube_hu[view_slice, :, :], cmap="gray")

# Now let's visualize the VOIs from the StructureSet.
for v, voi in enumerate(cst.vois):
    mask = sitk.GetArrayViewFromImage(voi.mask)
    plt.imshow(
        (v + 1) * mask[view_slice, :, :],
        cmap="jet",
        alpha=0.5 * (mask[view_slice, :, :] > 0),
        vmin=1,
        vmax=len(cst.vois),
    )

# Since we want to do planning, we create a plan object.
# We choose protons as the radiation mode and a Generic machine included with pyRadPlan
pln = PhotonPlan(machine="Generic")

# We use the StfGeneratorIMPT to create a simple beam configuration
# with a single beam at 0 degrees
stfgen = StfGeneratorPhotonIMRT(pln)
stfgen.bixel_width = 5.0
stfgen.gantry_angles = [0.0]

# We generate the beam geometry on the CT and CST
stf = stfgen.generate(ct, cst)

# Let's calculate the dose influence matrix
dij = calc_dose_influence(ct, cst, stf, pln)
