from typing import cast
import pytest

try:
    from importlib import resources  # Standard from Python 3.9+
except ImportError:
    import importlib_resources as resources  # Backport for older versions

import SimpleITK as sitk
import numpy as np

import pyRadPlan.io.matRad as matRadIO
from pyRadPlan.core import np2sitk
from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, StructureSet


@pytest.fixture
def tg119():
    path = resources.files("pyRadPlan.data.phantoms")
    tg119_file = path.joinpath("TG119.mat")
    tg119 = matRadIO.load(tg119_file)

    ct = validate_ct(tg119["ct"])
    cst = validate_cst(tg119["cst"], ct=ct)
    return {"ct": ct, "cst": cst}


def test_linear_index_to_world(tg119):
    ct = cast(CT, tg119["ct"])
    cst = cast(StructureSet, tg119["cst"])

    ct_array = sitk.GetArrayViewFromImage(ct.cube_hu)

    np_index = cst.vois[0].indices_numpy
    sitk_index = cst.vois[0].indices
    np_index_raveled = np.unravel_index(np_index, ct_array.shape)
    np_index_raveled = np.array(
        [np_index_raveled[0][0], np_index_raveled[1][0], np_index_raveled[2][0]]
    ).astype(np.int_)

    sitk_index = tuple([v.item() for v in np_index_raveled[::-1]])

    sitk_point = np.array(ct.cube_hu.TransformIndexToPhysicalPoint(sitk_index))

    np_coord = np2sitk.linear_indices_to_image_coordinates(
        indices=np_index[0], image=ct.cube_hu, index_type="numpy"
    )

    assert np.isclose(sitk_point, np_coord).all()
