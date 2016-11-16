from __future__ import division
from builtins import range
import math
import numpy as np
import scipy.linalg
# from . import DataConfig
from mdcgenutils.nearest_correlation import nearcorr


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

    return mass.astype(dtype=float)


def locate_centroids(clus_cfg):
    """
    TODO
    Args:
        clus_cfg (DataConfig): Configuration.

    Returns:
        np.array: Matrix (n_clusters, n_feats) with positions of centroids.
    """
    centroids = np.zeros((clus_cfg.n_clusters, clus_cfg.n_feats))

    # TODO understand idx
    p = 1.
    idx = 1
    for i, c in enumerate(clus_cfg._cmax):
        p *= c
        if p > 2 * clus_cfg.n_clusters + clus_cfg.outliers / clus_cfg.n_clusters:
            idx = i
            break

    locis = np.arange(p)
    np.random.shuffle(locis)
    clin = locis[:clus_cfg.n_clusters]

    # voodoo magic for obtaining centroids
    clin = np.array([clin] * idx).T
    first = ((clin % clus_cfg._cmax[:idx]) + 1 )/ (clus_cfg._cmax[:idx] + 1) \
            + ((np.random.rand(clus_cfg.n_clusters, idx) - 0.5) * np.array([clus_cfg.comp_factor] * idx).T)
    second = np.floor(clus_cfg._cmax[idx:] * np.random.rand(clus_cfg.n_clusters, clus_cfg.n_feats - idx) + 1) \
             / (clus_cfg._cmax[idx:] + 1) \
             + ((np.random.rand(clus_cfg.n_clusters, clus_cfg.n_feats - idx) - 0.5).T * clus_cfg.comp_factor).T
    centroids[:, :idx] = first.reshape((clus_cfg.n_clusters, idx))
    centroids[:, idx:] = second.reshape((clus_cfg.n_clusters, clus_cfg.n_feats - idx))

    return centroids, locis, idx


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
    mass = np.insert(mass, 0, clus_cfg.outliers)  # class 0 is now the outliers (this changes to -1 further down)
    mass /= mass.sum()

    labels = np.random.choice(clus_cfg.n_clusters + 1, n_samples, p=mass) - 1
    # label -1 corresponds to outliers
    data = np.zeros((n_samples, clus_cfg.n_feats))

    # generate samples for each cluster
    for label in range(clus_cfg.n_clusters):
        cluster = clus_cfg.clusters[label]
        indexes = (labels == label)
        samples = sum(indexes)  # nr of samples in this cluster
        if cluster.mv:
            for f in range(clus_cfg.n_feats):
                data[indexes, f] = cluster.distributions[f](samples, cluster.comp_factor)
        else:
            raise NotImplementedError('"mv" = False not implemented yet.')

        while True:
            # generate correlation matrix
            corr = np.ones((clus_cfg.n_feats, clus_cfg.n_feats))
            iu = np.triu_indices(len(corr), k=1)
            corr[iu] = cluster.corr * 2 * (np.random.rand(len(iu[0])) - 0.5)  # upper triangle
            corr.T[iu] = corr[iu]  # lower triangle

            # get valid correlation
            corrected_corr = nearcorr(corr)
            try:
                t_mat = np.linalg.cholesky(corrected_corr)
            except np.linalg.linalg.LinAlgError:
                print('oops...')
                continue
            data[indexes] = data[indexes].dot(t_mat)  # apply correlation to data
            break


        # compute random rotation
        if cluster.rotate:
            rot_mat = 2 * (np.random.rand(clus_cfg.n_feats, clus_cfg.n_feats) - 0.5)
            ort = scipy.linalg.orth(rot_mat)
            if ort.shape == rot_mat.shape:  # check if `rot_mat` is full rank, so that `ort` keeps the same shape
                data[indexes] = data[indexes].dot(ort)


        # add noisy variables
        for d in cluster.n_noise:
            data[indexes, d] = np.random.rand(samples)

        # add centroid
        data[indexes] += clus_cfg._centroids[label]

    # generate outliers
    indexes = (labels == -1)
    out = sum(indexes)

    # TODO make code more readable
    # voodoo magic for generating outliers
    locis = clus_cfg._locis[clus_cfg.n_clusters:]
    locis = np.array([locis[np.arange(out) % len(locis)]] * clus_cfg._idx).T
    first = (locis % clus_cfg._cmax[:clus_cfg._idx]) \
            / (clus_cfg._cmax[:clus_cfg._idx] + 1.) \
            + ((1. / (clus_cfg._cmax[:clus_cfg._idx] + 1)) * np.random.rand(out, clus_cfg._idx)
               - (1. / (2 * (clus_cfg._cmax[:clus_cfg._idx] + 1))))
    second = np.floor(clus_cfg._cmax[clus_cfg._idx:] * np.random.rand(out, clus_cfg.n_feats - clus_cfg._idx) + 1.) \
             / (clus_cfg._cmax[clus_cfg._idx:] + 1) \
             + ((1 / (clus_cfg._cmax[clus_cfg._idx:] + 1)) * np.random.rand(out, clus_cfg.n_feats - clus_cfg._idx)
                - (1. / (2 * (clus_cfg._cmax[clus_cfg._idx] + 1))))
    data[indexes,:clus_cfg._idx] = first.reshape((out, clus_cfg._idx))
    data[indexes,clus_cfg._idx:] = second.reshape((out, clus_cfg.n_feats - clus_cfg._idx))

    return data, labels
