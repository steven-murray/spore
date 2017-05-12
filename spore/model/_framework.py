import copy

class Component(object):
    def __init__(self, **model_params):
        # Check that all parameters passed are valid
        for k in model_params:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for the %s model" % (k, self.__class__.__name__))

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_params)