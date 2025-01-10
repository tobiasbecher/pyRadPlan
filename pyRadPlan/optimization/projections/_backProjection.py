#%% External package import

from numpy import array, array_equal
from abc import abstractmethod

#%% Class definition


class BackProjection:
    def __init__(self):

        self.__w_cache__ = array([])
        self.__w_grad_cache__ = array([])
        self.__d__ = array([])
        self.__w_grad__ = array([])

    def compute_dose(self, w):

        if not array_equal(self.__w_cache__, w):
            self.__d__ = self.compute_dose_result(w)
            self.__w_cache__ = w

        return self

    def compute_weight_gradient(self, dose_grad, w):

        if not array_equal(self.__w_grad_cache__, w):
            self.__w_grad__ = self.project_gradient(dose_grad, w)
            self.__w_grad_cache__ = w

        return self

    def get_dose_result(self):

        return self.__d__

    def get_weight_gradient(self):

        return self.__w_grad__

    def compute_dose_result(self, w):

        self.__d__ = self.ccompute_single_dose(w)

        return self.__d__

    def project_gradient(self, dose_grad, w):

        self.__w_grad__ = self.project_single_gradient(dose_grad, w)

        return self.__w_grad__

    @abstractmethod
    def ccompute_single_dose(self, w):
        pass

    @abstractmethod
    def project_single_gradient(self, dose_grad, w):
        pass
