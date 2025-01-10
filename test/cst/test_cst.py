import os
import pytest

import SimpleITK as sitk
import numpy as np

import pyRadPlan.io.matRad as matRadIO
from pyRadPlan.ct import create_ct


from pyRadPlan.ct import CT
from pyRadPlan.cst import (
    StructureSet,
    create_cst,
    validate_cst,
    create_voi,
    VOI,
    ExternalVOI,
    Target,
    OAR,
)


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


@pytest.fixture
def generic_ct():
    # Create a simple 3D CT image
    ct_array = np.zeros((10, 10, 10), dtype=np.float32)
    ct_image = sitk.GetImageFromArray(ct_array)
    ct_image.SetSpacing((1.0, 1.0, 1.0))
    ct_image.SetOrigin((0.0, 0.0, 0.0))
    ct_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return CT(cube_hu=ct_image)


@pytest.fixture
def generic_vois(generic_ct):
    # Create simple VOIs with different overlap priorities
    mask1 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask1[3:5, 3:5, 3:5] = 1
    mask_image1 = sitk.GetImageFromArray(mask1)
    mask_image1.CopyInformation(generic_ct.cube_hu)

    mask2 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask2[2:6, 2:6, 2:6] = 1
    mask_image2 = sitk.GetImageFromArray(mask2)
    mask_image2.CopyInformation(generic_ct.cube_hu)

    mask3 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask3[4:7, 4:7, 4:7] = 1
    mask_image3 = sitk.GetImageFromArray(mask3)
    mask_image3.CopyInformation(generic_ct.cube_hu)

    mask4 = np.zeros((10, 10, 10), dtype=np.uint8)
    mask4[1:8, 1:8, 1:8] = 1
    mask_image4 = sitk.GetImageFromArray(mask4)
    mask_image4.CopyInformation(generic_ct.cube_hu)

    voi1 = Target(name="CTV", mask=mask_image1, ct_image=generic_ct, overlap_priority=1)
    voi2 = Target(name="PTV", mask=mask_image2, ct_image=generic_ct, overlap_priority=2)
    voi3 = OAR(name="OAR", mask=mask_image3, ct_image=generic_ct)
    voi4 = ExternalVOI(name="BODY", mask=mask_image4, ct_image=generic_ct)

    return [voi1, voi2, voi3, voi4]


def test_apply_overlap_priorities(generic_ct, generic_vois):
    # Create a StructureSet with the VOIs
    structure_set = StructureSet(ct_image=generic_ct, vois=generic_vois)

    # Apply overlap priorities
    structure_set_overlap = structure_set.apply_overlap_priorities()

    # Check the resulting masks
    voi1_mask = sitk.GetArrayViewFromImage(structure_set.vois[0].mask)
    voi1_mask_overlapped = sitk.GetArrayViewFromImage(structure_set_overlap.vois[0].mask)

    voi2_mask = sitk.GetArrayViewFromImage(structure_set.vois[1].mask)
    voi2_mask_overlapped = sitk.GetArrayViewFromImage(structure_set_overlap.vois[1].mask)

    voi3_mask = sitk.GetArrayViewFromImage(structure_set.vois[2].mask)
    voi3_mask_overlapped = sitk.GetArrayViewFromImage(structure_set_overlap.vois[2].mask)

    voi4_mask = sitk.GetArrayViewFromImage(structure_set.vois[3].mask)
    voi4_mask_overlapped = sitk.GetArrayViewFromImage(structure_set_overlap.vois[3].mask)

    # VOI1 should have its mask unchanged
    assert np.isclose(voi1_mask, voi1_mask_overlapped).all()

    ol_mask = voi1_mask > 0

    # Check VOI 2 overlapped by VOI 1
    assert (voi2_mask_overlapped[ol_mask] == 0).all()
    assert (voi2_mask_overlapped[np.logical_and(voi2_mask > 0, ol_mask == 0)] == 1).all()
    ol_mask = np.logical_or(ol_mask, voi2_mask > 0)

    # Check VOI 3 overlapped by VOI 2 and VOI 1
    assert (voi3_mask_overlapped[voi1_mask > 0] == 0).all()
    assert (voi3_mask_overlapped[np.logical_and(voi2_mask > 0, ol_mask == 0)] == 1).all()
    ol_mask = np.logical_or(ol_mask, voi3_mask > 0)

    # Check VOI 4 overlapped by VOI 3, VOI 2 and VOI 1
    assert (voi4_mask_overlapped[ol_mask] == 0).all()
    assert (voi4_mask_overlapped[np.logical_and(voi3_mask > 0, ol_mask == 0)] == 1).all()


def test_apply_overlap_priorities_same_priority(generic_ct, generic_vois):
    for v in generic_vois:
        v.overlap_priority = 1

    # Create a StructureSet with the VOIs
    structure_set = StructureSet(ct_image=generic_ct, vois=generic_vois)

    # Apply overlap priorities
    structure_set_overlap = structure_set.apply_overlap_priorities()

    # Check the resulting masks
    for i in range(len(generic_vois)):
        voi_mask = sitk.GetArrayViewFromImage(structure_set.vois[i].mask)
        voi_mask_overlapped = sitk.GetArrayViewFromImage(structure_set_overlap.vois[i].mask)

        assert np.isclose(voi_mask, voi_mask_overlapped).all()
