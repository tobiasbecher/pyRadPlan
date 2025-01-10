#%% Internal package import

from ._backProjection import BackProjection

#%% Class definition


class ConstantRBEProjection(BackProjection):
    def __init__(self, dij, pln):

        self.dij = dij
        self.pln = pln

        super().__init__()

    def ccompute_single_dose(self, w):

        d = self.dij["physicalDose"] @ (self.pln["RBE"] * w)

        return d

    def project_single_gradient(self, dose_grad, w):

        w_grad = self.dij["physicalDose"].T @ (self.pln["RBE"] * dose_grad)

        return w_grad
