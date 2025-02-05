"""
Example of importing a patient from a MatRad file and visualizing the CT,
VOIs with matplotlib.

Also produces a simple beam configuration and visualizes the resulting radiological depth cubes by
calling the RayTracer.
"""

# Standard Library Imports
import logging

try:
    from importlib import resources  # Standard from Python 3.9+
except ImportError:
    import importlib_resources as resources  # Backport for older versions

# Third Party Imports from pyRadPlan requirements
import pymatreader
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# pyRadPlan Module Imports
from pyRadPlan.ct import validate_ct, default_hlut
from pyRadPlan.cst import validate_cst
from pyRadPlan.plan import IonPlan
from pyRadPlan.stf import StfGeneratorIMPT
from pyRadPlan.raytracer import RayTracerSiddon

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
pln = IonPlan(radiation_mode="protons", machine="Generic")

# We use the StfGeneratorIMPT to create a simple beam configuration
# with a single beam at 0 degrees
stfgen = StfGeneratorIMPT(pln)
stfgen.bixel_width = 5.0
stfgen.gantry_angles = [90.0]

# We generate the beam geometry on the CT and CST
stf = stfgen.generate(ct, cst)

# Now let's look at the water equivalent path length for our beam.
# For that, we use a default HU->rSP table to convert our CT to water-equivalent thickness and then
# call a voxel-wise RayTracing algorithm (proposed by Siddon) to calculate the radiological depth.
rt = RayTracerSiddon([ct.compute_wet(default_hlut())])
rad_depth_cubes = rt.trace_cubes(stf[0])

# We could write the image if we want, but we don't do this by default.
# sitk.WriteImage(ct.compute_wet(default_hlut()), "wet.nrrd")
# sitk.WriteImage(rad_depth_cubes[0], "rad_depth_cubes.nrrd")

# Let's see about our range of radiological depths
rd_np = sitk.GetArrayFromImage(rad_depth_cubes[0])
rd_np.ravel()[np.isnan(rd_np.ravel())] = 0
min_rad_depth = np.min(rd_np)
max_rad_depth = np.nanmax(rd_np)
print(f"Minimum rad depth: {min_rad_depth}Maximum rad depth: {max_rad_depth}")

# Now we do visualization of the corresponding slice of our WEPL cube
rad_depth_slice = rd_np[view_slice, :, :]

plt.imshow(
    rad_depth_slice,
    cmap="jet",
    alpha=0.5 * (rad_depth_slice > 0.0),
    vmin=min_rad_depth,
    vmax=max_rad_depth,
)

# And finally, output!
plt.draw()
plt.pause(0.001)
plt.show()
