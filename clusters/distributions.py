import numpy as np


def gap(n, param):
    out = np.random.normal(0, param, 2 * n)
    q25 = np.percentile(out, q=25.1)
    q75 = np.percentile(out, q=74.9)
    return out[(out < q25) + (out > q75)][:n]


def check_input(distributions):
    """
    Checks if the input distributions are valid. That is, check if they are either strings or functions. If they are
    strings, also check if they are contained in `distributions_list`.

    Args:
        distributions (list of list of (str or function)): Distributions given as input.

    Returns:
        (list of list of function): Functions for the distributions given as input.
    """
    return [[get_dist_function(d) for d in l] for l in distributions]


distributions_list = {
    'uniform': lambda n, param: np.random.uniform(-param, param, n),
    'gaussian': lambda n, param: np.random.normal(0, param, n),
    'logistic': lambda n, param: np.random.logistic(0, param, n),
    'triangular': lambda n, param: np.random.triangular(-param, 0, param, n),
    'gamma': lambda n, param: np.random.gamma(2 + 8 * np.random.rand(), param / 5, n),
    'gap': gap
}


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
