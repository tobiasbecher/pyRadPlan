import pytest

import numpy as np
import SimpleITK as sitk

from pyRadPlan import CT, StructureSet
from pyRadPlan import plot_slice


@pytest.fixture
def sample_ct():
    cube_hu = sitk.Image(100, 100, 10, sitk.sitkFloat32)
    return CT(cube_hu=cube_hu)


@pytest.fixture
def sample_cst(sample_ct):
    mask_np = np.zeros((10, 100, 100), dtype=np.uint8)
    mask_np[1:8, 20:80, 20:80] = 1
    mask = sitk.GetImageFromArray(mask_np)
    return StructureSet(
        vois=[{"mask": mask, "voi_type": "TARGET", "name": "test", "ct_image": sample_ct}],
        ct_image=sample_ct,
    )


@pytest.fixture
def sample_overlay():
    rand_image = np.random.rand(10, 100, 100).astype(np.float32)
    overlay = sitk.GetImageFromArray(rand_image)
    return overlay


def test_plot_slice_noargs():
    with pytest.raises(ValueError):
        plot_slice()


def test_plot_slice_ct(sample_ct):
    plot_slice(ct=sample_ct)


def test_plot_slice_cst(sample_cst):
    plot_slice(cst=sample_cst)


def test_plot_slice_ct_cst(sample_ct, sample_cst):
    plot_slice(ct=sample_ct, cst=sample_cst)


def test_plot_slice_with_overlay(sample_ct, sample_cst, sample_overlay):
    plot_slice(ct=sample_ct, cst=sample_cst, overlay=sample_overlay)


def test_plot_slice_parameters_coronal(sample_ct, sample_cst, sample_overlay):
    plot_slice(
        ct=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        view_slice=50,
        plane="coronal",
        overlay_unit="Gy",
        contour_line_width=2,
        overlay_alpha=0.2,
        overlay_rel_threshold=0.5,
    )


def test_plot_slice_parameters_sagittal(sample_ct, sample_cst, sample_overlay):
    plot_slice(
        ct=sample_ct,
        cst=sample_cst,
        overlay=sample_overlay,
        plane="sagittal",
    )


def test_plot_slice_parameters_invalid_plane(sample_ct, sample_cst, sample_overlay):
    with pytest.raises(ValueError):
        plot_slice(
            ct=sample_ct,
            cst=sample_cst,
            overlay=sample_overlay,
            plane="invalid_plane",
        )
