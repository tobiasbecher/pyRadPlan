from typing import Union
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.plan import IonPlan, PhotonPlan, validate_pln
from pyRadPlan.stf import (
    StfGeneratorIMPT,
    StfGeneratorPhotonIMRT,
    SteeringInformation,
    validate_stf,
)


def generate_stf(
    ct: Union[CT, dict], cst: Union[StructureSet, dict], pln: Union[StructureSet, dict]
) -> SteeringInformation:
    ct = validate_ct(ct)
    cst = validate_cst(cst, ct=ct)
    pln = validate_pln(pln)

    # TODO: obtain generator from pln.prop_stf
    if isinstance(pln, IonPlan):
        stfgen = StfGeneratorIMPT(pln)
    elif isinstance(pln, PhotonPlan):
        stfgen = StfGeneratorPhotonIMRT(pln)
    else:
        raise ValueError("Unknown plan!")

    return validate_stf(stfgen.generate(ct, cst))
