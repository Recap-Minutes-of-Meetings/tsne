import numpy as np
from scipy.spatial.distance import squareform


def get_distance_matrix(array, is_squareform=False):
    """Return matrix of Euclidean distances of points in array."""
    X = np.zeros((array.shape[0], array.shape[0]))
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            X[i][j] = np.sqrt(np.sum((array[i] - array[j]) ** 2))
    if is_squareform:
        return squareform(X)
    return X
