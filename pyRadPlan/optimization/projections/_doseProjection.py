#%% Internal package import

from ._backProjection import BackProjection

#%% Class definition


class DoseProjection(BackProjection):
    def __init__(self, dij, pln):

        self.dij = dij

        super().__init__()

    def ccompute_single_dose(self, w):

        d = self.dij["physicalDose"] @ w

        return d

    def project_single_gradient(self, dose_grad, w):

        w_grad = self.dij["physicalDose"].T @ dose_grad

        return w_grad
