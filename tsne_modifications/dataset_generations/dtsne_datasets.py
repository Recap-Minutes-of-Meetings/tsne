import numpy as np


def _dtsne_1():
    # Define centers of the 3 Gaussian clusters
    centers = np.array([[10, 0], [0, 15], [-10, 0]])
    n_points = 300

    # Define the scaling factors for the spread of the clusters
    scales = np.array([1, 2, 4])

    dataset = np.empty((0, 2))
    labels = np.empty((0, 1), dtype=int)

    for i in range(len(centers)):
        # Draw n_points from the ith Gaussian
        cluster = np.random.randn(n_points, 2) + centers[i]
        cluster *= scales[i]
        
        dataset = np.vstack((dataset, cluster))
        labels_i = np.full((n_points, 1), i, dtype=int)
        labels = np.vstack((labels, labels_i))

    return dataset, labels

def _dtsne_2():
    # Define centers of the 3 Gaussian clusters
    centers = np.array([[10, 0], [0, 15], [-10, 0]])

    # Define the number of points to draw from each Gaussian
    n_points = np.array([100, 200, 500])

    scale = 1
    dataset = np.empty((0, 2))
    labels = np.empty((0, 1), dtype=int)

    for i in range(len(centers)):
        # Draw n_points[i] from the ith Gaussian
        cluster = np.random.randn(n_points[i], 2) + centers[i]
        cluster *= scale

        # Add the scaled cluster to the dataset
        dataset = np.vstack((dataset, cluster))

        labels_i = np.full((n_points[i], 1), i, dtype=int)
        labels = np.vstack((labels, labels_i))

    return dataset, labels

def _dtsne_3():
    n_dims = 50

    # Define centers of the 3 Gaussian clusters
    centers = np.random.uniform(0, 50, size=(3, n_dims))

    # Define the number of points to draw from each Gaussian
    n_points = np.array([200, 400, 600])

    scale = 2

    dataset = np.empty((0, n_dims))
    labels = np.empty((0, 1), dtype=int)

    for i in range(len(centers)):
        # Draw n_points[i] from the ith Gaussian
        cluster = np.random.randn(n_points[i], n_dims) + centers[i]
        cluster *= scale

        # Add the scaled cluster to the dataset
        dataset = np.vstack((dataset, cluster))

        # Add labels for the ith Gaussian
        labels_i = np.full((n_points[i], 1), i, dtype=int)
        labels = np.vstack((labels, labels_i))

    return dataset, labels

def _dtsne_4():
    n_dims = 50

    # Define centers of the 3 Gaussian clusters
    centers = np.random.uniform(0, 50, size=(3, n_dims))
    n_points = np.array([300, 300, 300])

    # Define the scaling factor for the spread of the clusters
    scale = np.array([2, 4, 8])

    dataset = np.empty((0, n_dims))
    labels = np.empty((0, 1), dtype=int)

    # Generate the dataset
    for i in range(len(centers)):
        # Draw n_points[i] from the ith Gaussian
        cluster = np.random.randn(n_points[i], n_dims) + centers[i]
        cluster *= scale[i]

        dataset = np.vstack((dataset, cluster))

        labels_i = np.full((n_points[i], 1), i, dtype=int)
        labels = np.vstack((labels, labels_i))

    return dataset, labels

def _dtsne_5():
    n_dims = 50

    # Define centers of the 10 Gaussian clusters
    centers = np.random.uniform(0, 50, size=(10, n_dims))

    # Define the number of points to draw from each Gaussian
    n_points = 200

    # Define the scaling factor for the spread of the clusters
    scale = np.arange(1, 11)

    # Initialize the dataset
    dataset = np.empty((0, n_dims))
    labels = np.empty((0, 1), dtype=int)

    # Generate the dataset
    for i in range(len(centers)):
        # Draw n_points from the ith Gaussian
        cluster = np.random.randn(n_points, n_dims) + centers[i]

        # Scale the cluster by the scale factor
        cluster *= scale[i]

        # Add the scaled cluster to the dataset
        dataset = np.vstack((dataset, cluster))

        # Add labels for the ith Gaussian
        labels_i = np.full((n_points, 1), i, dtype=int)
        labels = np.vstack((labels, labels_i))

    return dataset, labels

def _dtsne_6():
    # Define the number of dimensions
    n_dims = 150

    # Define centers of the 10 uniform clusters
    centers = np.random.uniform(0, 50, size=(10, n_dims))

    # Define the number of points to draw from each cluster
    n_points = np.full((10,), 200)

    # Define the scaling factor for the spread of the clusters
    scale = np.arange(1, 11)

    # Initialize the dataset
    dataset = np.empty((0, n_dims))
    labels = np.empty((0, 1), dtype=int)

    # Generate the dataset
    for i in range(len(centers)):
        # Draw n_points[i] from the ith Uniform distribution
        cluster = np.random.uniform(centers[i], size=(n_points[i], n_dims))

        # Scale the cluster by the scale factor
        cluster *= scale[i]

        # Add the scaled cluster to the dataset
        dataset = np.vstack((dataset, cluster))

        # Add labels for the ith cluster
        labels_i = np.full((n_points[i], 1), i, dtype=int)
        labels = np.vstack((labels, labels_i))

    return dataset, labels

def get_dtsne_datasets():
    return {
        "2D_1": _dtsne_1(),
        "2D_2": _dtsne_2(),
        "G3-S": _dtsne_3(),
        "G3-D": _dtsne_4(),
        "G10-D": _dtsne_5(),
        "U5-D": _dtsne_6(),
    }
