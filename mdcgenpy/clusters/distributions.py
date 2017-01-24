from __future__ import division
from six import integer_types, string_types
import math
import numpy as np


distributions_list = {
    'uniform': lambda shape, param: np.random.uniform(-param, param, shape),
    'gaussian': lambda shape, param: np.random.normal(0, param, shape),
    'logistic': lambda shape, param: np.random.logistic(0, param, shape),
    'triangular': lambda shape, param: np.random.triangular(-param, 0, param, shape),
    'gamma': lambda shape, param: np.random.gamma(2 + 8 * np.random.rand(), param / 5, shape),
    'gap': lambda shape, param: gap(shape, param)
}
"""List of distributions for which you can just provide a string as input."""

# Aliases for distributions should be put here.
distributions_list['normal'] = distributions_list['gaussian']

valid_distributions = distributions_list.keys()
"""List of valid strings for distributions."""


def _validate_shape(shape):
    if not (hasattr(shape, '__iter__') and (len(shape) == 2 or len(shape) == 1))\
            and not isinstance(shape, integer_types):
        raise ValueError('Error! "shape" must be an integer or a tuple with size 2!')
    return True


def _validate_shape_intradistance(shape):
    if not (hasattr(shape, '__iter__') and len(shape) == 2):
        raise ValueError('Error! "shape" must be a tuple with size 2!')
    return True


def _aux_rms(mat):
    return np.sqrt((mat**2.).sum(1) / mat.shape[1]).reshape((mat.shape[0], 1))


def _intradistance_aux(shape):
    assert _validate_shape_intradistance(shape)
    out = np.random.rand(*shape) - 0.5
    out = math.sqrt(shape[1]) * out / _aux_rms(out)
    return out


class Distribution(object):
    def __init__(self, f, **kwargs):
        self.f = f
        self.kwargs = kwargs

    def __call__(self, shape, intra_distance, *args, **kwargs):
        new_kwargs = self.kwargs.copy()
        new_kwargs.update(kwargs)  # add keyword arguments given in __init__
        if intra_distance:
            assert _validate_shape_intradistance(shape)
            out = _intradistance_aux(shape)
            return out * self.f((shape[0], 1), *args, **new_kwargs)
        else:
            assert _validate_shape(shape)
            return self.f(shape, *args, **new_kwargs)


def gap(shape, param):
    out = np.zeros(shape)
    for j in range(shape[1]):
        new_shape = (2 * shape[0], 1)
        aux = np.random.normal(0, param, new_shape)
        med_aux = _aux_rms(aux)
        median = np.median(med_aux)
        out[:, j] = aux[med_aux > median][:shape[0]]
    return out


def check_input(distributions):
    """
    Checks if the input distributions are valid. That is, check if they are either strings or functions. If they are
    strings, also check if they are contained in `distributions_list`.

    Args:
        distributions (list of list of (str or function)): Distributions given as input.

    Returns:
        (list of list of function): Functions for the distributions given as input.
    """
    return [[get_dist_function(d) for d in l] if hasattr(l, '__iter__') and not isinstance(l, string_types) else get_dist_function(l)
            for l in distributions]


def get_dist_function(d):
    """
    Transforms distribution name into respective function.

    Args:
        d (str or function): Input distribution str/function.

    Returns:
        function: Actual function to compute the intended distribution.
    """
    if isinstance(d, Distribution):
        return d
    elif hasattr(d, '__call__'):
        return Distribution(d)
    elif isinstance(d, string_types):
        try:
            return Distribution(distributions_list[d])
        except KeyError:
            raise ValueError('Invalid distribution name "' + d + '". Available names are: ' +
                             ', '.join(distributions_list.keys()))
    else:
        raise ValueError('Invalid distribution input!')
    # TODO also check for None (and provide random choice)
