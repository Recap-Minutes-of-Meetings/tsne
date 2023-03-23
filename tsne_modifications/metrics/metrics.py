import numpy as np
from sklearn.neighbors import NearestNeighbors


def local_reconstruction_correlation(X, n_neighbors=100):
    """
    Calculates the correlation between high- and low-dimensional distances of each point
    with its 100 closest neighbors using local reconstruction.

    Parameters:
    X (array-like, shape (n_samples, n_features)): Input data.
    n_neighbors (int): Number of neighbors to use for the reconstruction.

    Returns:
    correlation (float): The correlation between high- and low-dimensional distances.
    """
    # Fit a nearest neighbors model to the input data
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)

    # Find the k nearest neighbors for each point
    distances, indices = knn.kneighbors(X)

    # Remove the first column (it's just the point itself)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Compute the pairwise distances between the points
    pairwise_distances = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))

    # Compute the pairwise distances between each point and its reconstructed neighbors
    reconstructed_distances = np.sqrt(
        ((X[:, np.newaxis] - X[indices]) ** 2).sum(axis=2)
    )

    # Compute the correlation between the two sets of distances
    correlation = np.corrcoef(
        pairwise_distances.flatten(), reconstructed_distances.flatten()
    )[0, 1]

    return correlation


def nos(X, embeddings):
    n = X.shape[0]
    results = np.zeros((n, n - 2))
    for i in range(n):
        x_idx = sorted(range(len(X[i])), key=lambda j: X[i][j])
        e_idx = sorted(range(len(embeddings[i])), key=lambda j: embeddings[i][j])
        for k in range(2, n):
            now = set(x_idx[1 : k + 1]).intersection(set(e_idx[1 : k + 1]))
            results[i][k - 2] = len(now) / (n * k)
    return [0] + list(np.sum(results, axis=0))
