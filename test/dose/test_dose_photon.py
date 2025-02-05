# Test_file which calls pyRadPlan/dose/calcPhotonDose_test.py


# ct, stf, pln, cst = [], [], {"radiationMode": "photons"}, []


# def test_PhotonPencilBeamSVDEngine():
#     from pyRadPlan.dose.engines import PhotonPencilBeamSVDEngine

#     engine = PhotonPencilBeamSVDEngine()
#     assert engine
#     assert engine.name != None
#     # assert isinstance(DoseEngineBase, PhotonPencilBeamSVDEngine)


# # this should be last since it goes through the whole process
# def test_calcPhotonDose():
#     assert pln["radiationMode"] == "photons"
#     dij = dose.calcPhotonDose_dev(ct, stf, pln, cst)
#     assert dij is not None
