import pytest
import numpy as np
from pyRadPlan.optimization.objectives import *

# cst =
# dij = {'numOfVoxels':27}

# Arrange
@pytest.fixture
def get_cst_dij():
    cst = {"PTV": {"resized_indices": [1, 2, 3]}}
    dij = {"numOfVoxels": 27}
    return (cst, dij)


def test_DoseUniformity_constructor(get_cst_dij):
    cst, dij = get_cst_dij
    doseUni = DoseUniformity(cst, dij)
    assert doseUni.name == "Dose Uniformity"
    assert doseUni.parameter_names == []
    assert doseUni.parameter_types == []
    assert doseUni.parameters == []
    assert doseUni.weight == 1.0
    assert doseUni.cst == cst
    assert doseUni.dij == dij


def test_DoseUniformity_getters_setters(get_cst_dij):
    cst, dij = get_cst_dij
    doseUni = DoseUniformity(cst, dij)

    # check non computing functions
    assert [] == doseUni.get_parameters()
    doseUni.set_parameters([1, 2, 3])
    assert [1, 2, 3] == doseUni.get_parameters()
    assert 1.0 == doseUni.get_weight()
    doseUni.set_weight(2.0)
    assert 2.0 == doseUni.get_weight()


def test_DoseUniformity_compute_objective(get_cst_dij):
    cst, dij = get_cst_dij
    doseUni = DoseUniformity(cst, dij)

    dose = np.array([1, 2, 3])
    struct = "PTV"
    assert np.abs(doseUni.compute_objective(dose, struct) - 1) < 1e-10


def test_DoseUniformity_compute_gradient(get_cst_dij):
    cst, dij = get_cst_dij
    doseUni = DoseUniformity(cst, dij)
    dose = np.array([1, 2, 3])
    struct = "PTV"
    grad_expected = np.zeros(27)
    grad_expected[1:4] = 1 / 2 * np.array([-1, 0, 1])
    assert np.all((doseUni.compute_gradient(dose, struct) - grad_expected) < 1e-10)


def test_SquaredDeviation_constructor(get_cst_dij):
    cst, dij = get_cst_dij
    sqDev = SquaredDeviation(cst, dij, 2)
    assert sqDev.name == "Squared Deviation"
    assert sqDev.parameter_names == ["d^{ref}"]
    assert sqDev.parameter_types == ["dose"]
    assert sqDev.parameters == 2.0
    assert sqDev.weight == 1.0
    assert sqDev.cst == cst
    assert sqDev.dij == dij


def test_SquaredDeviation_getters_setters(get_cst_dij):
    cst, dij = get_cst_dij
    sqDev = SquaredDeviation(cst, dij, 2)
    # check non computing functions
    assert 2.0 == sqDev.get_parameters()
    sqDev.set_parameters(3.0)
    assert 3.0 == sqDev.get_parameters()
    assert 1.0 == sqDev.get_weight()
    sqDev.set_weight(2.0)
    assert 2.0 == sqDev.get_weight()


def test_SquaredDeviation_compute_objective(get_cst_dij):
    cst, dij = get_cst_dij
    dose = np.array([1, 2, 3])
    struct = "PTV"
    sqDev = SquaredDeviation(cst, dij, 2)
    assert sqDev.compute_objective(dose, struct) == 2 / 3


def test_SquaredDeviation_compute_gradient(get_cst_dij):
    cst, dij = get_cst_dij
    dose = np.array([1, 2, 3])
    struct = "PTV"
    sqDev = SquaredDeviation(cst, dij, 2)
    grad_expected = np.zeros(27)
    grad_expected[1:4] = 2 / 3 * np.array([-1, 0, 1])
    assert np.all(sqDev.compute_gradient(dose, struct) == grad_expected)


def test_SquaredOverdosing_constructor(get_cst_dij):
    cst, dij = get_cst_dij
    sqOver = SquaredOverdosing(cst, dij, 2)
    assert sqOver.name == "Squared Overdosing"
    assert sqOver.parameter_names == ["d^{max}"]
    assert sqOver.parameter_types == ["dose"]
    assert sqOver.parameters == 2.0
    assert sqOver.weight == 1.0
    assert sqOver.cst == cst
    assert sqOver.dij == dij


def test_SquaredOverdosing_getters_setters(get_cst_dij):
    cst, dij = get_cst_dij
    sqOver = SquaredOverdosing(cst, dij, 2)
    # check non computing functions
    assert 2.0 == sqOver.get_parameters()
    sqOver.set_parameters(3.0)
    assert 3.0 == sqOver.get_parameters()
    assert 1.0 == sqOver.get_weight()
    sqOver.set_weight(2.0)
    assert 2.0 == sqOver.get_weight()


def test_SquaredOverdosing_compute_objective(get_cst_dij):
    cst, dij = get_cst_dij
    sqOver = SquaredOverdosing(cst, dij, 2)
    dose = np.array([1, 2, 3])
    struct = "PTV"
    assert sqOver.compute_objective(dose, struct) == 1 / 3


def test_SquaredOverdosing_compute_gradient(get_cst_dij):
    cst, dij = get_cst_dij
    dose = np.array([1, 2, 3])
    struct = "PTV"
    sqOver = SquaredOverdosing(cst, dij, 2)
    grad_expected = np.zeros(27)
    grad_expected[1:4] = 2 / 3 * np.array([0, 0, 1])
    assert np.all(sqOver.compute_gradient(dose, struct) == grad_expected)


def test_SquaredUnderdosing_constructor(get_cst_dij):
    cst, dij = get_cst_dij
    sqUnder = SquaredUnderdosing(cst, dij, 2)
    assert sqUnder.name == "Squared Underdosing"
    assert sqUnder.parameter_names == ["d^{min}"]
    assert sqUnder.parameter_types == ["dose"]
    assert sqUnder.parameters == 2.0
    assert sqUnder.weight == 1.0
    assert sqUnder.cst == cst
    assert sqUnder.dij == dij


def test_SquaredUnderdosing_getters_setters(get_cst_dij):
    cst, dij = get_cst_dij
    sqUnder = SquaredUnderdosing(cst, dij, 2)
    # check non computing functions
    assert 2.0 == sqUnder.get_parameters()
    sqUnder.set_parameters(3.0)
    assert 3.0 == sqUnder.get_parameters()
    assert 1.0 == sqUnder.get_weight()
    sqUnder.set_weight(2.0)
    assert 2.0 == sqUnder.get_weight()


def test_SquaredUnderdosing_compute_objective(get_cst_dij):
    cst, dij = get_cst_dij
    sqUnder = SquaredUnderdosing(cst, dij, 2)
    dose = np.array([1, 2, 3])
    struct = "PTV"
    assert sqUnder.compute_objective(dose, struct) == 1 / 3


def test_SquaredUnderdosing_compute_gradient(get_cst_dij):
    cst, dij = get_cst_dij
    dose = np.array([1, 2, 3])
    struct = "PTV"
    sqUnder = SquaredUnderdosing(cst, dij, 2)
    grad_expected = np.zeros(27)
    grad_expected[1:4] = 2 / 3 * np.array([-1, 0, 0])
    assert np.all(sqUnder.compute_gradient(dose, struct) == grad_expected)


def test_EUD_constructor(get_cst_dij):
    cst, dij = get_cst_dij
    eud = EUD(cst, dij, 0, 3)
    assert eud.name == "EUD"
    assert eud.parameter_names == ["EUD^{ref}", "k"]
    assert eud.parameter_types == ["dose", "numeric"]
    assert eud.parameters == [0.0, 3.0]
    assert eud.weight == 1.0
    assert eud.cst == cst
    assert eud.dij == dij


def test_EUD_getters_setters(get_cst_dij):
    cst, dij = get_cst_dij
    eud = EUD(cst, dij, 0, 3)
    # check non computing functions
    assert [0.0, 3.0] == eud.get_parameters()
    eud.set_parameters([3.0, 4.0])
    assert [3.0, 4.0] == eud.get_parameters()
    assert 1.0 == eud.get_weight()
    eud.set_weight(2.0)
    assert 2.0 == eud.get_weight()


def test_EUD_compute_objective(get_cst_dij):
    cst, dij = get_cst_dij
    eud = EUD(cst, dij, 0, 3)
    dose = np.array([1, 2, 3])
    struct = "PTV"
    assert (
        eud.compute_objective(dose, struct) - (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 6
    ) < 1e-10


def test_EUD_compute_gradient(get_cst_dij):
    cst, dij = get_cst_dij
    eud = EUD(cst, dij, 0, 3)
    dose = np.array([1, 2, 3])
    struct = "PTV"
    grad_expected = np.zeros(27)
    dEUd = (1 + 2 ** (1 / 3) + 3 ** (1 / 3)) ** 2 * np.array([1, 2, 3]) ** (-2 / 3) * 1 / 3**3
    EUd = (1 / 3 * (1 + 2 ** (1 / 3) + 3 ** (1 / 3))) ** 3
    grad_expected[1:4] = 2 * (EUd - 0) * dEUd
    assert np.all((eud.compute_gradient(dose, struct) - grad_expected) < 1e-10)


def test_MeanDose_constructor(get_cst_dij):
    cst, dij = get_cst_dij
    meanDose = MeanDose(cst, dij, 2)
    assert meanDose.name == "Mean Dose"
    assert meanDose.parameter_names == ["d^{ref}"]
    assert meanDose.parameter_types == ["dose"]
    assert meanDose.parameters == 2.0
    assert meanDose.weight == 1.0
    assert meanDose.cst == cst
    assert meanDose.dij == dij


def test_MeanDose_getters_setters(get_cst_dij):
    cst, dij = get_cst_dij
    meanDose = MeanDose(cst, dij, 2)
    # check non computing functions
    assert 2.0 == meanDose.get_parameters()
    meanDose.set_parameters(3.0)
    assert 3.0 == meanDose.get_parameters()
    assert 1.0 == meanDose.get_weight()
    meanDose.set_weight(2.0)
    assert 2.0 == meanDose.get_weight()


def test_MeanDose_compute_objective(get_cst_dij):
    cst, dij = get_cst_dij
    meanDose = MeanDose(cst, dij, 2)
    dose = np.array([1, 2, 3])
    struct = "PTV"
    assert meanDose.compute_objective(dose, struct) == 0


def test_MeanDose_compute_gradient(get_cst_dij):
    cst, dij = get_cst_dij
    meanDose = MeanDose(cst, dij, 2)
    dose = np.array([1, 2, 3])
    struct = "PTV"
    grad_expected = np.zeros(27)
    print(meanDose.compute_gradient(dose, struct))
    assert np.all(meanDose.compute_gradient(dose, struct) == grad_expected)


def test_MinDVH_constructor(get_cst_dij):
    cst, dij = get_cst_dij
    minDVH = MinDVH(cst, dij, 2, 3)
    assert minDVH.name == "Min DVH"
    assert minDVH.parameter_names == ["d", "V^{min}"]
    assert minDVH.parameter_types == ["dose", "numeric"]
    assert minDVH.parameters == [2.0, 3.0]
    assert minDVH.weight == 1.0
    assert minDVH.cst == cst
    assert minDVH.dij == dij


def test_MinDVH_getters_setters(get_cst_dij):
    cst, dij = get_cst_dij
    minDVH = MinDVH(cst, dij, 2, 3)
    # check non computing functions
    assert [2.0, 3.0] == minDVH.get_parameters()
    minDVH.set_parameters([3.0, 4.0])
    assert [3.0, 4.0] == minDVH.get_parameters()
    assert 1.0 == minDVH.get_weight()
    minDVH.set_weight(2.0)
    assert 2.0 == minDVH.get_weight()


def test_MinDVH_compute_objective():
    cst = {"PTV": {"resized_indices": np.arange(0, 100, 1)}}
    dij = {"numOfVoxels": 1000}
    minDVH = MinDVH(cst, dij, 30, 0.95)
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    struct = "PTV"
    assert minDVH.compute_objective(dose, struct) == 841
    assert minDVH.compute_objective(dose_2, struct) == 0


def test_MinDVH_compute_gradient():
    cst = {"PTV": {"resized_indices": np.arange(0, 100, 1)}}
    dij = {"numOfVoxels": 1000}
    minDVH = MinDVH(cst, dij, 30, 0.95)
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    struct = "PTV"
    grad_expected = np.zeros(1000)
    grad_expected[0:100] = np.ones(100) * -0.58
    grad_expected2 = np.zeros(1000)
    assert np.all(minDVH.compute_gradient(dose, struct) == grad_expected)
    assert np.all(minDVH.compute_gradient(dose_2, struct) == grad_expected2)


def test_MaxDVH_constructor():
    cst = {"PTV": {"resized_indices": np.arange(0, 100, 1)}}
    dij = {"numOfVoxels": 1000}
    maxDVH = MaxDVH(cst, dij, 30, 0.95)
    assert maxDVH.name == "Max DVH"
    assert maxDVH.parameter_names == ["d", "V^{max}"]
    assert maxDVH.parameter_types == ["dose", "numeric"]
    assert maxDVH.parameters == [30.0, 0.95]
    assert maxDVH.weight == 1.0
    assert maxDVH.cst == cst
    assert maxDVH.dij == dij


def test_MaxDVH_getters_setters():
    cst = {"PTV": {"resized_indices": np.arange(0, 100, 1)}}
    dij = {"numOfVoxels": 1000}
    maxDVH = MaxDVH(cst, dij, 30, 0.95)
    # check non computing functions
    assert [30.0, 0.95] == maxDVH.get_parameters()
    maxDVH.set_parameters([3.0, 4.0])
    assert [3.0, 4.0] == maxDVH.get_parameters()
    assert 1.0 == maxDVH.get_weight()
    maxDVH.set_weight(2.0)
    assert 2.0 == maxDVH.get_weight()


def test_MaxDVH_compute_objective():
    cst = {"PTV": {"resized_indices": np.arange(0, 100, 1)}}
    dij = {"numOfVoxels": 1000}
    maxDVH = MaxDVH(cst, dij, 30, 0.95)
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    struct = "PTV"
    assert maxDVH.compute_objective(dose, struct) == 0
    assert maxDVH.compute_objective(dose_2, struct) == 400


def test_MaxDVH_compute_gradient():
    cst = {"PTV": {"resized_indices": np.arange(0, 100, 1)}}
    dij = {"numOfVoxels": 1000}
    maxDVH = MaxDVH(cst, dij, 30, 0.95)
    dose = np.ones(100)
    dose_2 = np.ones(100) * 50
    struct = "PTV"
    grad_expected = np.zeros(1000)
    grad_expected2 = np.zeros(1000)
    grad_expected2[0:100] = np.ones(100) * 0.4
    assert np.all(maxDVH.compute_gradient(dose, struct) == grad_expected)
    assert np.all(maxDVH.compute_gradient(dose_2, struct) == grad_expected2)
