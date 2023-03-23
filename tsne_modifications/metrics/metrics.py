import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.spatial import cKDTree


def global_spearman_rank(X, Y):
    # Compute pairwise distances between points in X and Y
    dx = squareform(pdist(X))
    dy = squareform(pdist(Y))

    # Compute the Spearman rank correlation between the distances
    rho, pval = spearmanr(dx.ravel(), dy.ravel())
    return rho


def local_reconstruction_correlation(X, Y, n_neighbors=100):
    """
    Calculates the correlation between high- and low-dimensional distances of each point
    with its 100 closest neighbors using local reconstruction.

    Parameters:
    X (array-like, shape (n_samples, n_features)): Input data.
    Y (array-like, shape (n_samples, n_features)): Input data.
    n_neighbors (int): Number of neighbors to use for the reconstruction.

    Returns:
    correlation (float): The correlation between high- and low-dimensional distances.
    """

    res = []
    for a in [X, Y]:
        dists = squareform(pdist(a))
        ind = np.argsort(dists, axis=1)[:, 1:n_neighbors + 1]
        res.append(np.take_along_axis(dists, ind, axis=1).flatten())

    rho, pval = spearmanr(res[0], res[1])
    return rho


def relative_density_reconstruction(X, Y, n_neighbors=100):
    """
    correlation between radii of balls enclosing the 100 neighbours of each point 
    in high- and low-dimensional space.
    """
    res = []
    for a in [X, Y]:
        dists = squareform(pdist(a))
        ind = np.argsort(dists, axis=1)[:, 1:n_neighbors + 1]
        res.append(np.take_along_axis(
            dists, ind, axis=1).max(axis=1).flatten())

    rho, pval = spearmanr(res[0], res[1])
    return rho


def nos(X, embeddings):
    n = X.shape[0]
    results = np.zeros((n, n - 2))
    for i in range(n):
        x_idx = sorted(range(len(X[i])), key=lambda j: X[i][j])
        e_idx = sorted(
            range(len(embeddings[i])), key=lambda j: embeddings[i][j])
        for k in range(2, n):
            now = set(x_idx[1: k + 1]).intersection(set(e_idx[1: k + 1]))
            results[i][k - 2] = len(now) / (n * k)
    return [0] + list(np.sum(results, axis=0))
