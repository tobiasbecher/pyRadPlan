from pyRadPlan.stf import StfGeneratorBase


class StfGeneratorMatradBase(StfGeneratorBase):
    def __init__(self, ct, cst, pln):
        self.propStf = pln["propStf"]

    def generate(self):
        pass
