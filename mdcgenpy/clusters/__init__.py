from numbers import Number
import random
import math
import six
import numpy as np
from . import distributions as dist
from . import generate


class ClusterGenerator(object):
    """
    Structure to handle the input and create clusters according to it.
    """
    def  __init__(self, seed=1, n_samples=2000, n_feats=2, k=5, min_samples=0, possible_distributions=None,
                  distributions=None, mv=True, corr=0., compactness_factor=0.1, alpha_n=1,
                  scale=True, outliers=50, rotate=True, add_noise=0, n_noise=None, ki_coeff=3., **kwargs):
        """
        Args:
            seed (int): Seed for the generation of random values. Useful for consistency.
            n_samples (int): Number of samples to generate.
            n_feats (int): Number of dimensions/features for each sample.
            k (int or list of int): Number of clusters to generate. If input is a list, each element in it specifies the
                number of samples in each cluster. In that case, the number of clusters will be the length of the list.
            min_samples (int): Minimum number of samples in each cluster. If 0, the default minimum for a cluster with
                :math:`N` samples is :math:`N/(\\text{ki_coeff}*k)`.
            possible_distributions (list): List of distributions to randomly choose from. Each element in this list
                must either be a valid str (valid str are defined in :data:`~.distributions.valid_distributions`
                OR a function which implements the distribution OR an instance of
                :class:`~.distributions.Distribution`.

                This parameter is overridden by ``distributions``, when set.
            distributions (str or function or .distributions.Distribution or list): Distribution to be used.
                If list, its length must be ``k``, and each element in the list must either be a valid str (indicating
                the distribution to be used) OR a function which implements the distribution OR a list of str/functions
                with length ``n_feats``.

                Instances of :class:`~.distributions.Distribution` can also be used.

                Valid str are defined in :data:`~.distributions.valid_distributions`.
            mv (bool or list of bool or None): Multivariate distributions or distributions defining intra-distances. If
                True, distributions define feature values (multivariate). If False, distributions define
                intra-distances.

                If None, this choice is made at random.

                If a list, its length must be ``k``, and each value in the list applies to one cluster.
            corr (float or list of float): Maximum (in absolute value) correlation between variables.

                If a list, its length must be ``k``, and each value in the list applies to one cluster.
            compactness_factor (float or list of float): Compactness factor.

                If a list, its length must be ``k``, and each value in the list applies to one cluster.
            alpha_n (float or list of float): Determines grid hyperplanes. If :math:`\\alpha_n > 0`, the number of
                hyperplanes is a factor of
                :math:`\\alpha_n * \\left \\lfloor{1 + \\frac{k}{\\log(k)}}\\right \\rfloor`.

                If :math:`\\alpha_n < 0`, the number of hyperplanes is :math:`|\\alpha_n|`.

                If a list, its length must be ``n_feats``, and each value in the list applies to one dimension.
            scale (bool or list of bool): Optimizes cluster separation based on grid size. If True, scale based on min
                distance between grid hyperplanes. If False, scale based on max distance between grid hyperplanes.

                If None, does not scale.

                If a list, its length must be ``k``, and each value in the list applies to one cluster.
            outliers (int): Number of outliers.
            rotate (bool or list of bool): If True, clusters can rotate.

                If a list, its length must be ``k``, and each value in the list applies to one cluster.
            add_noise (int): Add this number of noisy dimensions.
            n_noise (list): Parameter that manages noisy dimensions.

                If a list of int (of size :math:`\\leq` ``n_feats``, and each element is :math:`\\geq 0` and
                :math:`<` ``n_feats``), each dimension listed (0-indexed) will have only noise.

                If a list of list of int (of length ``k``, and each element is a list of length :math:`\\leq`
                ``n_feats``, with values :math:`\\geq 0` and :math:`<` ``n_feats``), each list indicates the noisy
                dimensions for a particular cluster.
            ki_coeff (float): Coefficient used to define the default minimum number of samples per cluster.
        """
        self.seed = seed
        self.n_samples = n_samples
        self.n_feats = n_feats
        self.k = k
        self.n_clusters = len(k) if type(k) == list else k
        self.min_samples = min_samples
        self.possible_distributions = possible_distributions if possible_distributions is not None \
            else ['gaussian', 'uniform']
        self.distributions = distributions
        self.mv = mv
        self.corr = corr
        self.compactness_factor = compactness_factor
        self.alpha_n = alpha_n
        self._cmax = None
        self.scale = scale
        self.outliers = outliers
        self.rotate = rotate
        self.add_noise = add_noise
        self.n_noise = n_noise if n_noise is not None else []
        self.ki_coeff = ki_coeff

        random.seed(self.seed)

        for key, val in kwargs.items():
            self.__dict__[key] = val

        self._distributions = None
        self._validate_parameters()
        self.clusters = self.get_cluster_configs()

        self._mass = None
        self._centroids = None
        self._locis = None
        self._idx = None

    def generate_data(self, batch_size=0):
        np.random.seed(self.seed)
        self._mass = generate.generate_mass(self)
        self._centroids, self._locis, self._idx = generate.locate_centroids(self)
        batches = generate.generate_clusters(self, batch_size)
        if batch_size == 0:  # if batch_size == 0, just return the data instead of the generator
            return next(batches)
        else:
            return batches

    def get_cluster_configs(self):
        return [Cluster(self, i) for i in range(self.n_clusters)]

    def _validate_parameters(self):
        """
        Method to validate the parameters of the object.
        """
        if hasattr(self.k, '__iter__'):
            if len(self.k) == 1:  # if only one input, no point in being a list
                self.k = self.k[0]
                self.n_clusters = self.k
            elif len(self.k) < 1:
                raise ValueError('"k" parameter must have at least one value!')
            else:
                if sum(self.k) != self.n_samples:
                    raise ValueError('Total number of points must be the same as the sum of points in each cluster!')

        if self.distributions is not None:
            # check validity of self.distributions, and turning it into a (n_clusters, n_feats) matrix
            if hasattr(self.distributions, '__iter__') and not type(self.distributions) == str:
                if len(self.distributions) != self.n_clusters:
                    raise ValueError('There must be exactly one distribution input for each cluster!')
                if hasattr(self.distributions[0], '__iter__'):
                    if not all(hasattr(elem, '__iter__') and len(elem) == self.n_feats for elem in self.distributions):
                        raise ValueError('Invalid distributions input! Input must have dimensions (n_clusters, n_feats).')
            else:
                self.distributions = [self.distributions] * self.n_clusters
            self._distributions = dist.check_input(self.distributions)
        else:
            self.distributions = [random.choice(self.possible_distributions) for _ in range(self.n_clusters)]
            self._distributions = dist.check_input(self.distributions)

        # check validity of self.mv, and turn it into a list with self.n_clusters elements
        if hasattr(self.mv, '__iter__'):
            if len(self.mv) != self.n_clusters:
                raise ValueError('There must be exactly one "mv" parameter for each cluster!')
        else:
            if self.mv is None:
                self.mv = [random.choice([True, False]) for _ in range(self.n_clusters)]
            else:
                self.mv = [self.mv] * self.n_clusters
        assert all(_validate_mv(elem) for elem in self.mv)


        # check validity of self.scale, and turn it into a list with self.n_clusters elements
        if hasattr(self.scale, '__iter__'):
            if len(self.scale) != self.n_clusters:
                raise ValueError('There must be exactly one "scale" parameter for each cluster!')
        else:
            self.scale = [self.scale] * self.n_clusters
        assert all(_validate_scale(elem) for elem in self.scale)

        # check validity of self.corr, and turn it into a list with self.n_clusters elements
        if hasattr(self.corr, '__iter__'):
            if len(self.corr) != self.n_clusters:
                raise ValueError('There must be exactly one correlation "corr" value for each cluster!')
        else:
            self.corr = [self.corr] * self.n_clusters
        assert all(_validate_corr(elem) for elem in self.corr)

        # check validity of self.alpha_n, and turn it into a list with self.n_feats elements
        if hasattr(self.alpha_n, '__iter__'):
            if len(self.alpha_n) != self.n_feats:
                raise ValueError('There must be exactly one hyperplane parameter "alpha_n" value for each dimension!')
        else:
            self.alpha_n = [self.alpha_n] * self.n_feats
        assert all(_validate_alpha_n(elem) for elem in self.alpha_n)

        # set self._cmax
        self._cmax = [math.floor(1 + self.n_clusters / math.log(self.n_clusters))] * self.n_feats \
            if self.n_clusters > 1 else [1 + 2 * (self.outliers > 1)] * self.n_feats
        self._cmax = [round(-a) if a < 0 else round(c * a) for a, c in zip(self.alpha_n, self._cmax)]
        self._cmax = np.array(self._cmax)

        # check validity of self.compactness_factor, and turn it into a list with self.n_clusters elements
        if hasattr(self.compactness_factor, '__iter__'):
            if len(self.compactness_factor) != self.n_clusters:
                raise ValueError('There must be exactly one compactness "compactness_factor" value for each cluster!')
        else:
            self.compactness_factor = [self.compactness_factor] * self.n_clusters
        assert all(_validate_compactness_factor(elem) for elem in self.compactness_factor)

        cmax_max = max(self._cmax)
        cmax_min = min(self._cmax)
        self.compactness_factor = [cp / cmax_max if s else (cp / cmax_min if not s else cp)
                                   for cp, s in zip(self.compactness_factor, self.scale)]

        # check validity of self.rotate, and turn it into a list with self.n_clusters elements
        if hasattr(self.rotate, '__iter__'):
            if len(self.rotate) != self.n_clusters:
                raise ValueError('There must be exactly one rotate value for each cluster!')
        else:
            self.rotate = [self.rotate] * self.n_clusters
        assert all(_validate_rotate(elem) for elem in self.rotate)

        # check validity of self.add_noise and self.n_noise
        if not isinstance(self.add_noise, six.integer_types):
            raise ValueError('Invalid input for "add_noise"! Input must be integer.')
        if hasattr(self.n_noise, '__iter__'):
            if len(self.n_noise) == 0:
                self.n_noise = [[]] * self.n_clusters
            if hasattr(self.n_noise[0], '__iter__'):
                if len(self.n_noise) != self.n_clusters:
                    raise ValueError('Invalid input for "n_noise"! List length must be the number of clusters.')
            else:
                self.n_noise = [self.n_noise] * self.n_clusters
        else:
            raise ValueError('Invalid input for "n_noise"! Input must be a list.')
        assert all(_validate_n_noise(elem, self.n_feats) for elem in self.n_noise)

    @property
    def mass(self):
        return self._mass


class Cluster(object):
    """
    Contains the parameters of an individual cluster.
    """

    settables = ['distributions', 'mv', 'corr', 'compactness_factor', 'scale', 'rotate', 'n_noise']
    """
    List of settable properties of Cluster. These are the parameters which can be set at a cluster level, and override
    the parameters of the cluster generator.
    """

    def __init__(self, cfg, idx, corr_matrix=None):
        """
        Args:
            cfg (ClusterGenerator): Configuration of the data.
            idx (int): Index of a cluster.
            corr_matrix (np.array): Valid correlation matrix to use in this cluster.
        """
        self.cfg = cfg
        self.idx = idx
        self.corr_matrix = corr_matrix

    def generate_data(self, samples):
        if hasattr(self.distributions, '__iter__'):
            out = np.zeros((samples, self.cfg.n_feats))
            for f in range(self.cfg.n_feats):
                out[:,f] = self.distributions[f](samples, self.mv, self.compactness_factor)
            return out
        else:
            return self.distributions((samples, self.cfg.n_feats), self.mv, self.compactness_factor)

    @property
    def n_feats(self):
        return self.cfg.n_feats

    @property
    def distributions(self):
        return self.cfg._distributions[self.idx]

    @distributions.setter
    def distributions(self, value):
        if isinstance(value, six.string_types):
            self.cfg._distributions[self.idx] = dist.get_dist_function(value)
        elif hasattr(value, '__iter__'):
            self.cfg._distributions[self.idx] = [dist.get_dist_function(d) for d in value]
        else:
            self.cfg._distributions[self.idx] = dist.get_dist_function(value)

    @property
    def mv(self):
        return self.cfg.mv[self.idx]

    @mv.setter
    def mv(self, value):
        assert _validate_mv(value)
        self.cfg.mv[self.idx] = value

    @property
    def corr(self):
        return self.cfg.corr[self.idx]

    @corr.setter
    def corr(self, value):
        assert _validate_corr(value)
        self.cfg.corr[self.idx] = value

    @property
    def compactness_factor(self):
        return self.cfg.compactness_factor[self.idx]

    @compactness_factor.setter
    def compactness_factor(self, value):
        assert _validate_compactness_factor(value)
        self.cfg.compactness_factor[self.idx] = value

    @property
    def scale(self):
        return self.cfg.scale[self.idx]

    @scale.setter
    def scale(self, value):
        assert _validate_scale(value)
        self.cfg.scale[self.idx] = value

    @property
    def rotate(self):
        return self.cfg.rotate[self.idx]

    @rotate.setter
    def rotate(self, value):
        assert _validate_rotate(value)
        self.cfg.rotate[self.idx] = value

    @property
    def n_noise(self):
        return self.cfg.n_noise[self.idx]

    @n_noise.setter
    def n_noise(self, value):
        assert _validate_n_noise(value, self.cfg.n_feats)
        self.cfg.n_noise[self.idx] = value


class ScheduledClusterGenerator(ClusterGenerator):
    """
    This cluster generator takes a schedule and all the ClusterGenerator arguments, and activates only the specified
    clusters in the schedule, for each time step.
    A time step is defined as one get call to ``self.mass``, which is done when generating each new batch.
    That is, one time step is one call to :func:`.generate.compute_batch`.
    """
    def __init__(self, schedule, *args, **kwargs):
        """
        Args:
            schedule (list): List in which each element contains the indexes of the clusters active in the respective
                time step.
            *args: args for :meth:`ClusterGenerator.__init__`.
            **kwargs: kwargs for :meth:`ClusterGenerator.__init__`.
        """
        super(ScheduledClusterGenerator, self).__init__(*args, **kwargs)
        self.cur_time = 0
        self.schedule = schedule

    @property
    def mass(self):
        mass = self._mass.copy()
        cur_clusters = self.schedule[self.cur_time % len(self.schedule)]
        for c in range(len(mass)):  # set the mass of clusters not scheduled now to 0
            if c not in cur_clusters:
                mass[c] = 0

        self.cur_time += 1  # increase time
        return mass


def _validate_mv(mv):
    """
    Checks validity of input for `mv`.

    Args:
        mv (bool): Input to check validity

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if mv not in [True, None, False]:
        raise ValueError('Invalid input value for "mv"!')
    return True


def _validate_corr(corr):
    """
    Checks validity of input for `corr`.

    Args:
        corr (float): Input to check validity.

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if not isinstance(corr, Number):
        raise ValueError('Invalid input value for "corr"! Values must be numeric')
    if not 0 <= corr <= 1:
        raise ValueError('Invalid input value for "corr"! Values must be between 0 and 1.')
    return True


def _validate_compactness_factor(compactness_factor):
    """
    Checks validity of input for `compactness_factor`.

    Args:
        compactness_factor (float): Input to check validity.

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if not isinstance(compactness_factor, Number):
        raise ValueError('Invalid input value for "compactness_factor"! Values must be numeric')
    # TODO 0 <= compactness_factor <= 1 ?
    return True


def _validate_alpha_n(alpha_n):
    """
    Checks validity of input for `alpha_n`.

    Args:
        alpha_n (float): Input to check validity.

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if not isinstance(alpha_n, Number):
        raise ValueError('Invalid input for "alpha_n"! Values must be numeric.')
    if alpha_n == 0:
        raise ValueError('Invalid input for "alpha_n"! Values must be different from 0.')
    return True


def _validate_scale(scale):
    """
    Checks validity of input for `scale`.

    Args:
        scale (bool): Input to check validity.

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if scale not in [True, None, False]:
        raise ValueError('Invalid input value for "scale"! Input must be boolean (or None).')
    return True


def _validate_rotate(rotate):
    """
    Checks validity of input for `rotate`.
    Args:
        rotate (bool): Input to check validity.

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if rotate not in [True, False]:
        raise ValueError('Invalid input for "rotate"! Input must be boolean.')
    return True


def _validate_n_noise(n_noise, n_feats):
    """
    Checks validity of input for `n_noise`.

    Args:
        n_noise (list of int): Input to check validity.
        n_feats (int): Number of dimensions/features.

    Returns:

    """
    if not hasattr(n_noise, '__iter__'):
        raise ValueError('Invalid input for "n_noise"! Input must be a list.')
    if len(n_noise) > n_feats:
        raise ValueError('Invalid input for "n_noise"! Input has more dimensions than total number of dimensions.')
    if not all(isinstance(n, six.integer_types) for n in n_noise):
        raise ValueError('Invalid input for "n_noise"! Input dimensions must be integers.')
    if not all(0 <= n < n_feats for n in n_noise):
        raise ValueError('Invalid input for "n_noise"! Input dimensions must be in the interval [0, "n_feats"[.')
    return True
