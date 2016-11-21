from __future__ import division
from six import integer_types, string_types
import math
import numpy as np


def _validate_shape(shape):
    if not (hasattr(shape, '__iter__') and (len(shape) == 2 or len(shape) == 1))\
            and not isinstance(shape, integer_types):
        raise ValueError('Error! "shape" must be an integer or a tuple with size 2!')
    return True


def _validate_shape_intradistance(shape):
    if not (hasattr(shape, '__iter__') and len(shape) == 2):
        raise ValueError('Error! "shape" must be a tuple with size 2!')
    return True


def gap(shape, param):
    assert _validate_shape(shape)
    new_shape = (2 * shape[0], shape[1]) if len(shape) == 2 else 2 * shape
    try:
        aux = np.zeros(shape)
        for f in range(new_shape[1]):
            out = np.random.normal(0, param, new_shape[0])
            q25 = np.percentile(out, q=25)
            q75 = np.percentile(out, q=75)
            aux[:, f] = out[(out < q25) + (out > q75)][:shape[0]]
        return aux
    except IndexError:
        out = np.random.normal(0, param, new_shape[0])
        q25 = np.percentile(out, q=25)
        q75 = np.percentile(out, q=75)
        return out[(out < q25) + (out > q75)][:shape[0]]


def _aux_rms(mat):
    return np.sqrt((mat**2.).sum(1) / mat.shape[1]).reshape((mat.shape[0], 1))


def gap_intradistance(shape, param):
    assert _validate_shape_intradistance(shape)
    aux = _intradistance_aux((2 * shape[0], shape[1]))
    out = np.random.normal(0, param, (2 * shape[0], 1)) * aux
    med_aux = _aux_rms(out)
    median = np.median(med_aux)
    return out[med_aux.reshape((med_aux.shape[0],)) > median][:shape[0]]


def _intradistance_aux(shape):
    assert _validate_shape_intradistance(shape)
    out = np.random.rand(*shape) - 0.5
    out = math.sqrt(shape[1]) * out / _aux_rms(out)
    return out


def uniform_intradistance(shape, param):
    out = _intradistance_aux(shape)
    return out * np.random.uniform(-param, param, shape[0]).reshape((shape[0], 1))


def gaussian_intradistance(shape, param):
    out = _intradistance_aux(shape)
    return out * np.random.normal(0, param, shape[0]).reshape((shape[0], 1))


def logistic_intradistance(shape, param):
    out = _intradistance_aux(shape)
    return out * np.random.logistic(0, param, shape[0]).reshape((shape[0], 1))


def triangular_intradistance(shape, param):
    out = _intradistance_aux(shape)
    return out * np.random.triangular(-param, 0, param, shape[0]).reshape((shape[0], 1))


def gamma_intradistance(shape, param):
    out = _intradistance_aux(shape)
    return out * np.random.gamma(2 + 8 * np.random.rand(), param / 5, shape).reshape((shape[0], 1))


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


distributions_list = {
    'uniform': lambda shape, param: np.random.uniform(-param, param, shape),
    'uniform_intradistance': lambda shape, param: uniform_intradistance(shape, param),
    'gaussian': lambda shape, param: np.random.normal(0, param, shape),
    'gaussian_intradistance': lambda shape, param: gaussian_intradistance(shape, param),
    'logistic': lambda shape, param: np.random.logistic(0, param, shape),
    'logistic_intradistance': lambda shape, param: logistic_intradistance(shape, param),
    'triangular': lambda shape, param: np.random.triangular(-param, 0, param, shape),
    'triangular_intradistance': lambda shape, param: triangular_intradistance(shape, param),
    'gamma': lambda shape, param: np.random.gamma(2 + 8 * np.random.rand(), param / 5, shape),
    'gamma_intradistance': lambda shape, param: gamma_intradistance(shape, param),
    'gap': lambda shape, param: gap(shape, param),
    'gap_intradistance': lambda shape, param: gap_intradistance(shape, param)
}
# aliases
distributions_list['normal'] = distributions_list['gaussian']
distributions_list['normal_intradistance'] = distributions_list['gaussian_intradistance']


def get_dist_function(d):
    """
    Transforms distribution name into respective function.

    Args:
        d (str or function): Input distribution str/function.

    Returns:
        function: Actual function to compute the intended distribution.
    """
    if hasattr(d, '__call__'):
        return d
    elif type(d) == str:
        try:
            return distributions_list[d]
        except KeyError:
            raise ValueError('Invalid distribution name "' + d + '". Available names are: ' +
                             ', '.join(distributions_list.keys()))
    else:
        raise ValueError('Invalid distribution input!')
    # TODO also check for None (and provide random choice)
