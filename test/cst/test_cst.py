import os
import pytest

import SimpleITK as sitk
import numpy as np

import pyRadPlan.io.matRad as matRadIO
from pyRadPlan.ct import create_ct


from pyRadPlan.cst import StructureSet, create_cst, validate_cst, create_voi


def test_cst_from_matrad_mat_file(matrad_import):

    ct = create_ct(matrad_import["ct"])
    cst = create_cst(matrad_import["cst"], ct=ct)
    assert isinstance(cst, StructureSet)

    cst = validate_cst(matrad_import["cst"], ct=ct)
    assert isinstance(cst, StructureSet)

    with pytest.raises(ValueError):
        cst = create_cst(matrad_import["cst"])

    with pytest.raises(ValueError):
        cst = validate_cst(matrad_import["cst"])


def test_cst_to_matrad(matrad_import, tmpdir):
    ct = create_ct(matrad_import["ct"])
    cst = create_cst(matrad_import["cst"], ct=ct)

    matrad_list = cst.to_matrad()
    assert isinstance(matrad_list, list)

    tmp_mat_path = os.path.join(tmpdir, "test_cst.mat")
    matRadIO.save(tmp_mat_path, {"cst": matrad_list})
    assert os.path.exists(tmp_mat_path)

    tmp = matRadIO.load(tmp_mat_path)

    assert isinstance(tmp, dict)
    assert isinstance(tmp["cst"], list)


def test_cst_target_voxels(generic_input_3d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d

    mask_3d_2 = mask_3d.copy()
    mask_3d_2.fill(0)
    mask_3d_2[0, 0, 0] = 1

    mask_3d_3 = mask_3d.copy()
    mask_3d_3.fill(0)
    mask_3d_3[1, 1, 1] = 1

    mask_3d_4 = mask_3d.copy()
    mask_3d_4.fill(0)
    mask_3d_4[0, 1, 0] = 1

    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_3d_2 = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d_2)
    voi_3d_3 = create_voi(voi_type="OAR", name=name_3d, ct_image=ct_3d, mask=mask_3d_3)
    voi_3d_4 = create_voi(voi_type="EXTERNAL", name=name_3d, ct_image=ct_3d, mask=mask_3d_4)

    cst = StructureSet(vois=[voi_3d, voi_3d_2, voi_3d_3, voi_3d_4], ct_image=ct_3d)

    index_union = cst.target_union_voxels()
    assert (index_union == cst.target_union_voxels(order="sitk")).all()
    index_union_np = cst.target_union_voxels(order="numpy")
    mask_union = cst.target_union_mask()

    assert (index_union == np.array([0, 5000])).all()
    assert (index_union_np == np.array([0, 1])).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="F")[index_union] == 1).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="C")[index_union_np] == 1).all()


def test_cst_patient_voxels(generic_input_3d):
    name_3d, ct_3d, mask_3d, _, _ = generic_input_3d

    mask_3d_3 = mask_3d.copy()
    mask_3d_3.fill(0)
    mask_3d_3[0, 0, 0] = 1

    mask_3d_4 = mask_3d.copy()
    mask_3d_4.fill(0)
    mask_3d_4[-1, -1, -1] = 1

    voi_3d = create_voi(voi_type="TARGET", name=name_3d, ct_image=ct_3d, mask=mask_3d)
    voi_3d_3 = create_voi(voi_type="OAR", name=name_3d, ct_image=ct_3d, mask=mask_3d_3)
    voi_3d_4 = create_voi(voi_type="OAR", name=name_3d, ct_image=ct_3d, mask=mask_3d_4)

    cst = StructureSet(vois=[voi_3d, voi_3d_3, voi_3d_4], ct_image=ct_3d)

    index_union = cst.patient_voxels()
    assert (index_union == cst.patient_voxels(order="sitk")).all()
    index_union_np = cst.patient_voxels(order="numpy")
    mask_union = cst.patient_mask()

    assert (index_union == np.array([0, 5000, mask_3d_3.size - 1])).all()
    assert (index_union_np == np.array([0, 1, mask_3d_3.size - 1])).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="F")[index_union] == 1).all()
    assert (sitk.GetArrayViewFromImage(mask_union).ravel(order="C")[index_union_np] == 1).all()

    voi_3d_4 = create_voi(voi_type="EXTERNAL", name=name_3d, ct_image=ct_3d, mask=mask_3d_4)
    cst.vois[-1] = voi_3d_4
    index_union = cst.patient_voxels()
    assert (index_union == cst.patient_voxels(order="sitk")).all()
    index_union_np = cst.patient_voxels(order="numpy")
    mask_union = cst.patient_mask()

    assert (index_union == np.array([mask_3d_4.size - 1])).all()
    assert (index_union_np == np.array([mask_3d_4.size - 1])).all()


def test_target_center_of_mass():
    ct = create_ct(cube_hu=sitk.Image(10, 10, 10, sitk.sitkInt16))
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1
    mask[3, 3, 3] = 1

    voi = create_voi(voi_type="TARGET", name="test", ct_image=ct, mask=mask)
    cst = StructureSet(vois=[voi], ct_image=ct)

    com = cst.target_center_of_mass()
    assert np.allclose(com, np.array([1.5, 1.5, 1.5]))

    image4d = sitk.JoinSeries([sitk.Image(10, 10, 10, sitk.sitkInt16) for _ in range(3)])
    ct = create_ct(cube_hu=image4d)
    mask = np.zeros((3, 10, 10, 10), dtype=np.uint8)
    mask[0, 0, 0, 0] = 1
    mask[0, 1, 1, 1] = 1
    mask[0, 2, 2, 2] = 1
    mask[0, 3, 3, 3] = 1

    voi = create_voi(voi_type="TARGET", name="test", ct_image=ct, mask=mask)
    cst = StructureSet(vois=[voi], ct_image=ct)

    com = cst.target_center_of_mass()
    assert np.allclose(com, np.array([1.5, 1.5, 1.5]))
