#%% External package import

import abc
from collections.abc import Iterable

#%% Class definition


class ObjectiveClass(metaclass=abc.ABCMeta):
    def flatten(iterable):

        for it in iterable:
            if isinstance(it, Iterable) and not isinstance(it, (str, bytes)):
                yield from ObjectiveClass.flatten(it)
            else:
                yield it

    @abc.abstractproperty
    def name(self):
        pass

    @abc.abstractproperty
    def parameter_names(self):
        pass

    @abc.abstractproperty
    def parameter_types(self):
        pass

    @abc.abstractproperty
    def parameters(self):
        pass

    @abc.abstractproperty
    def weight(self):
        pass

    @abc.abstractmethod
    def compute_objective(self, *args):
        pass

    @abc.abstractmethod
    def compute_gradient(self, *args):
        pass

    @abc.abstractmethod
    def get_parameters(self):
        pass

    @abc.abstractmethod
    def set_parameters(self, *args):
        pass

    @abc.abstractmethod
    def get_weight(self):
        pass

    @abc.abstractmethod
    def set_weight(self, *args):
        pass

    @staticmethod
    def _check_objective(self, name, parameter_names, parameter_types, parameters, weight):

        # Check objective name
        if not isinstance(name, str):
            raise TypeError('Variable "name" must be string')

        # Check objective parameter_names
        if not isinstance(parameter_names, (tuple, list)):
            raise TypeError('Variable "parameter_names" must be a tuple or a list')
        else:
            if not all([isinstance(el, str) for el in ObjectiveClass.flatten(parameter_names)]):
                raise TypeError('Elements in "parameter_names" must be string')

        # Check objective parameter_types
        if not isinstance(parameter_types, (tuple, list)):
            raise TypeError('Variable "parameter_types" must be a tuple or a list')
        else:
            if not all([isinstance(el, str) for el in ObjectiveClass.flatten(parameter_types)]):
                raise TypeError('Elements in "parameter_names" must be string')

        # Check objective parameters
        if (not isinstance(parameters, (tuple, list))) and (
            not isinstance(parameters, (int, float))
        ):
            raise TypeError(
                'Variable "parameters" must either be a tuple, a list or a single numeric value'
            )
        elif (isinstance(parameters, (tuple, list))) and (
            not all([isinstance(el, (int, float)) for el in ObjectiveClass.flatten(parameters)])
        ):
            raise TypeError('Elements in "parameters" must be numeric')

        # Check objective penalty
        if not isinstance(weight, (int, float)):
            raise TypeError('Variable "weight" must be numeric')
        return
