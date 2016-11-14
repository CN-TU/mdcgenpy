from __future__ import division
from builtins import range
from functools import reduce
import operator
import math
import random
import numpy as np
import scipy.linalg
from . import DataConfig
from ..utils.nearest_correlation import nearcorr


def generate_mass(clus_cfg):
    """
    TODO
    Args:
        clus_cfg (DataConfig): Configuration

    Returns:
        np.array: Array with len == nr of clusters, where each entry is the number of samples in the corresponding
            to generate in the corresponding cluster.
    """
    if type(clus_cfg.k) == list:
        mass = np.array(clus_cfg.k)
    else:
        mass = np.random.uniform(0, 1, clus_cfg.n_clusters)
        total_mass = mass.sum()
        mass = np.vectorize(math.floor)(clus_cfg.n_samples * mass / total_mass)
        abs_mass = mass.sum()
        if abs_mass < clus_cfg.n_samples:  # if samples are unassigned, send them to the cluster with least samples
            min_ind = np.argmin(mass)
            mass[min_ind] += clus_cfg.n_samples - abs_mass

        # guarantee there are enough samples in each cluster
        if clus_cfg.min_samples <= 0:
            min_mass = round(clus_cfg.n_samples / (clus_cfg.ki_coeff * clus_cfg.n_clusters))
        else:
            min_mass = clus_cfg.min_samples
        need_to_add = True
        while need_to_add:
            need_to_add = False
            min_ind = np.argmin(mass)
            if mass[min_ind] < min_mass:
                max_ind = np.argmax(mass)
                extra = min_mass - mass[min_ind]
                mass[max_ind] -= extra
                mass[min_ind] += extra
                need_to_add = True

    return mass


def locate_centroids(clus_cfg):
    """
    TODO
    Args:
        clus_cfg (DataConfig): Configuration.

    Returns:
        np.array: Matrix (n_clusters, n_feats) with positions of centroids.
    """
    centroids = np.zeros(clus_cfg.n_clusters, clus_cfg.n_feats)

    # TODO don't quite understand what this does
    p = 1.
    idx = None
    for i, c in enumerate(clus_cfg._cmax):
        p *= c
        if p > 2 * clus_cfg.n_clusters + clus_cfg.outliers / clus_cfg.n_clusters:
            idx = i
            break
    assert idx != None

    # TODO understand this variable name
    locis = random.shuffle(range(reduce(operator.mul, clus_cfg._cmax[:idx])))
    clin = locis[:clus_cfg.n_clusters]

    for i in range(clus_cfg.n_clusters):
        res = clin[i]
        for j in range(idx):
            centroids[i, j] = res % clus_cfg._cmax[j]
            res = math.floor(res / clus_cfg._cmax[j])
            centroids[i, j] /= clus_cfg._cmax[j] + 1
            centroids[i, j] += (np.random.rand() - 0.5) * clus_cfg.comp_factor[i]
        assert idx < clus_cfg.n_feats  # TODO this is here because similar is in Felix's code; remove when this is understood
        for j in range(idx, clus_cfg.n_feats):
            centroids[i, j] = math.floor(clus_cfg._cmax[j] * np.random.rand() + 1) / (clus_cfg._cmax[j] + 1)
            centroids += (np.random.rand() - 0.5) * clus_cfg.comp_factor[i]

    return centroids, locis


def generate_clusters(clus_cfg, batch_size = 0):
    """
    TODO
    Args:
        clus_cfg (DataConfig): Configuration.
        batch_size (int): Number of samples for each batch.

    Yields:

    """
    if batch_size == 0:
        batch_size = clus_cfg.n_samples
    for batch in range(((clus_cfg.n_samples - 1) // batch_size) + 1):
        n_samples = min(batch_size, clus_cfg.n_samples - batch * batch_size)
        data, labels = compute_batch(clus_cfg, n_samples)
        yield data, labels


def compute_batch(clus_cfg, n_samples):
    """
    TODO
    Args:
        clus_cfg (DataConfig): Configuration.
        n_samples (int): Number of samples in the batch.

    Returns:
        np.array: Generated sample.
    """
    # get probabilities of each class
    mass = clus_cfg._mass
    mass.prepend(clus_cfg.outliers)  # class 0 is now the outliers (this changes to -1 later)
    mass /= sum(mass)

    labels = np.random.choice(clus_cfg.n_clusters, n_samples, mass) - 1
    # label -1 corresponds to outliers
    data = np.zeros(n_samples, clus_cfg.n_feats)

    # generate samples for each cluster
    for label in range(clus_cfg.n_clusters):
        cluster = clus_cfg.clusters[label]
        indexes = (labels == label)
        samples = sum(indexes)  # nr of samples in this cluster
        if cluster.mv:
            for f in clus_cfg.n_feats:
                data[indexes, f] = cluster.distributions[f](samples, cluster.comp_factor)
        else:
            raise NotImplementedError('"mv" = False not implemented yet.')

        # generate correlation matrix
        corr = np.ones((clus_cfg.n_feats, clus_cfg.n_feats))
        iu = np.triu_indices(len(corr), k=1)
        corr[iu] = cluster.corr * 2 * (np.random.rand(iu[0].shape) - 0.5)  # upper triangle
        corr.T[iu] = corr[iu]  # lower triangle

        # get valid correlation
        corrected_corr = nearcorr(corr)
        t_mat = np.linalg.cholesky(corrected_corr)
        data[indexes] = data[indexes].dot(t_mat)  # apply correlation to data


        # compute random rotation
        if cluster.rotate:
            rot_mat = 2 * (np.random.rand((clus_cfg.n_feats, clus_cfg.n_feats)) - 0.5)
            ort = scipy.linalg.orth(rot_mat)
            if ort.shape == rot_mat:  # check if `rot_mat` is full rank, so that `ort` keeps the same shape
                data[indexes] = data[indexes].dot(ort)


        # add noisy variables
        for d in cluster.n_noise:
            data[indexes, d] = np.random.rand(samples)

        # add centroid
        data[indexes] += clus_cfg._centroids[label]

    # generate outliers
    indexes = (labels == -1)
    max_val = 1.1 * np.max(data[~indexes])
    min_val = 1.1 * np.min(data[~indexes])
    out = len(indexes)
    # s = np.zeros((out, clus_cfg.n_feats))

    # TODO understand this
    # TODO "vectorize" loop
    for i in range(out):
        res = clus_cfg._locis[(i % (clus_cfg.n_clusters - len(clus_cfg._locis))) + clus_cfg.n_clusters]
        s = np.zeros(clus_cfg.n_feats)
        aux = np.zeros(clus_cfg.n_feats)
        for j in range(len(clus_cfg._locis)):
            s[j] = res % clus_cfg._cmax[j]
            res = math.floor(res / clus_cfg._cmax[j])
            s[j] /= clus_cfg._cmax[j] + 1.
            aux[j] = (1. / (clus_cfg._cmax[j] + 1)) * np.random.rand() - (1. / (2 * (clus_cfg._cmax[j] + 1)))
        for j in range(len(clus_cfg._locis), clus_cfg.n_feats):
            s[j] = math.floor(clus_cfg._cmax[j] * np.random.rand() + 1.) / (clus_cfg._cmax[j] + 1)
            aux[j] = (1 / (clus_cfg._cmax + 1)) * np.random.rand() - (1. / (2 * (clus_cfg[j] + 1)))
        data[indexes[i]] = s + aux

    return data, labels
