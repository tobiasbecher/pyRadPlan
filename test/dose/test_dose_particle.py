# test for pyRadPlan.dose
import numpy as np
import pytest
from pyRadPlan.ct import CT
import SimpleITK as sitk
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine, get_available_engines, get_engine


@pytest.fixture
def sample_cst():  # TODO: not necessary atm
    cst = {"example": None}
    return cst


@pytest.fixture
def sample_stf_dict():
    stf = {
        "gantry_angle": 0,
        "couch_angle": 0,
        "bixel_width": 1,
        "radiation_mode": "protons",
        "machine": "Generic",
        "sad": 100000,
        "iso_center": [0, 0, 0],
        "num_of_rays": 1,
        "ray": {},  # TODO: quite complex structure.
        "source_point_bev": [0, -10000, 0],
        "source_point": [0, 0, 0],
        "num_of_bixels_per_ray": [0, 0, 0],
        "longitudinal_spot_spacing": 1,
        "total_number_of_bixels": 1,
    }
    return stf


@pytest.fixture
def sample_stf():
    beam1 = {
        "gantry_angle": 0,
        "couch_angle": 0,
        "bixel_width": 1,
        "radiation_mode": "protons",
        "machine": "Generic",
        "sad": 100000,
        "iso_center": [0, 0, 0],
        "num_of_rays": 1,
        "ray": {},  # TODO: quite complex structure.
        "source_point_bev": [0, -10000, 0],
        "source_point": [0, 0, 0],
        "num_of_bixels_per_ray": [0, 0, 0],
        "longitudinal_spot_spacing": 1,
        "total_number_of_bixels": 1,
    }
    beam2 = beam1
    beam2["gantry_angle"] = 90
    # beam2["radiation_mode"] = "Photons"
    stf = SteeringInformation(beams=[beam1, beam2])
    return stf


@pytest.fixture
def sample_ct_dict():
    ct = {
        "cube": np.zeros((167, 167, 129)),
        "resolution": {},
        "x": np.zeros(167),  # fill in the rest of the array
        "y": np.zeros(167),
        "z": np.zeros(129),
        "cubeDim": [167, 167, 129],
        "numOfCtScen": 1,
        "dicomInfo": {
            "PixelSpacing": [0.9766, 0.9766],
            "SlicePositions": np.zeros(129),
            "SliceThickness": np.zeros(129),
            "ImagePositionPatient": [-250, -250, -160],
            "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
            "PatientPosition": "HFS",
            "Width": 512,
            "Height": 512,
            "RescaleSlope": 1,
            "RescaleIntercept": -1024,
            "Manufacturer": "GE MEDICAL SYSTEMS",
            "ManufacturerModelName": "LightSpeed RT",
            "ConvolutionKernel": "STANDARD",
            "PatientName": {},  # TODO: not necessary atm
        },
        "dicomMeta": {},  # TODO: not necessary atm. quite complex
        "timeStamp": "22-Jan-2018 14:17:42",
        "cubeHU": np.zeros((167, 167, 129)),
        "hlut": np.array(
            [
                [-1024, 0.0032],
                [200, 1.2000],
                [449, 1.2000],
                [2000, 2.4907],
                [2048, 2.5306],
                [3071, 2.5306],
            ]
        ),
    }
    # ct = validate_ct(ct)
    return ct


@pytest.fixture
def sample_ct():
    """Create a sample SimpleITK image for testing."""
    sample_array = np.random.rand(50, 100, 100) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    # np_image = sitk.GetArrayViewFromImage(image)
    # np_image_t = np_image.reshape((50, 100, 100),order='F')

    ct = CT(cube_hu=image)
    return ct


@pytest.fixture
def sample_pln():
    pln = {
        "radiationMode": "protons",  # either photons / protons / carbon
        "machine": "Generic",
        "numOfFractions": 30.0,
        "propStf": {
            # beam geometry settings
            "bixelWidth": 5.0,
            "gantryAngles": [0.0, 90.0, 180.0, 270.0],
            "couchAngles": [0.0, 0.0, 0.0, 0.0],
            "numOfBeams": 4.0,
            "isoCenter": [
                [251.3089, 236.4147, 162.6421],
                [251.3089, 236.4147, 162.6421],
                [251.3089, 236.4147, 162.6421],
                [251.3089, 236.4147, 162.6421],
            ],
        },
        # dose calculation settings
        "propDoseCalc": {
            "doseGrid": {
                "resolution": {
                    "x": 3.0,
                    "y": 3.0,
                    "z": 3.0,
                },
            },
        },
        # optimization settings
        "propOpt": {
            "optimizer": "IPOPT",
            "bioOptimization": "none",
            "runDAO": False,
            "runSequencing": True,
        },
    }
    # pln = validate_pln(pln)
    return pln


# class DummyDoseEngine(DoseEngineBase):
#     def __init__(self):
#         self.name = "DummyDoseEngine"

#     def calc_dose(self, ct, stf, pln, cst):
#         ct = validate_ct(ct)

#         dij = init_dose_calc(ct, stf, cst)
#         return return


def test_available_engines(sample_pln):
    engines = get_available_engines(sample_pln)
    assert isinstance(engines, dict)
    assert len(engines.items()) > 0


def test_get_engine(sample_pln):
    sample_pln["propDoseCalc"]["engine"] = "HongPB"
    engine = get_engine(sample_pln)
    assert engine
    assert isinstance(engine, ParticleHongPencilBeamEngine)


def test_get_engine_default(sample_pln):
    engine = get_engine(sample_pln)
    assert engine
    assert isinstance(engine, ParticleHongPencilBeamEngine)


def test_get_engine_invalid(sample_pln):
    sample_pln["propDoseCalc"]["engine"] = "InvalidEngine"
    with pytest.raises(ValueError):
        engine = get_engine(sample_pln)


def test_ParticleHongPencilBeamEngine(sample_pln):
    engine = ParticleHongPencilBeamEngine(sample_pln)
    assert engine
    assert engine.name != None
    # assert isinstance(DoseEngineBase, ParticleHongPencilBeamEngine)


# Testing CalcDoseInit ------- currently not working (cst has to be expanded)
# TODO: split this test into multiple tests
# def test_calcDoseInit(sample_stf, sample_ct, sample_pln, sample_cst):
#     from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine
#     from pyRadPlan.dose.calcDoseInit import init_dose_calc

#     engine = ParticleHongPencilBeamEngine()
#     engine.mult_scen = "nomScen"
#     dij = init_dose_calc(engine, sample_ct, sample_cst, sample_stf)

#     # Test if create_scenario_model is called correctly
#     assert isinstance(engine.mult_scen, ScenarioModel)

#     # TODO: Test if Biomodel is called correctly
#     # assert ...

#     assert dij is not None


# this tests the whole procedure
# def test_calcParticleDose(sample_ct, sample_stf, sample_cst):
#     from pyRadPlan.dose.calcParticleDose_dev import calcParticleDose_dev

#     dij = calcParticleDose_dev(sample_ct, sample_stf, sample_cst)

#     assert dij is not None
