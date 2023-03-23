import numpy as np


def synthesize_8_4_4(size, seed=42):
    np.random.seed(seed)
    center = np.random.normal(0, 2, size=(4, 8))

    array = np.zeros((size, 14))

    tmp = np.zeros_like(array[:, :8])
    wrong_answers = []
    for i in range(size):
        cluster = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])
        tmp[i] = center[cluster - 1] + np.random.normal(0, cluster / 20, size=8)
        wrong_answers.append(cluster)
    array[:, :8] = tmp

    center = np.random.normal(0, 2, size=(4, 4))
    tmp = np.zeros_like(array[:, 8:12])
    answers = []
    for i in range(size):
        cluster = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])
        tmp[i] = center[cluster - 1] + np.random.normal(0, cluster / 20, size=4)
        answers.append(cluster)
    array[:, 8:12] = tmp

    array[:, 12:] = np.random.normal(0, 0.1, size=(size, 2))
    return array, answers, wrong_answers


def synthesize_4_2_2(size, seed=42):
    np.random.seed(seed)
    center = np.eye(4)

    array = np.zeros((size, 10))

    tmp = np.zeros_like(array[:, :4])
    wrong_answers = []
    for i in range(size):
        cluster = np.random.choice(4)
        tmp[i] = center[cluster] + np.random.normal(0, 0.01, size=4)
        wrong_answers.append(cluster + 1)
    array[:, :4] = tmp

    center = np.array([[1 / 3, 0], [0, 1 / 3]])
    tmp = np.zeros_like(array[:, 4:6])
    answers = []
    for i in range(size):
        cluster = np.random.choice(2)
        tmp[i] = center[cluster] + np.random.normal(0, 0.01, size=2)
        answers.append(cluster + 1)
    array[:, 4:6] = tmp

    array[:, 6:] = np.random.normal(0, 0.01, size=(size, 4))
    return array, answers, wrong_answers


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


def get_tdsne_datasets():
    return [
        _dtsne_1(),
        _dtsne_2(),
        _dtsne_3(),
        _dtsne_4(),
        _dtsne_5(),
        _dtsne_6(),
    ]


def synthesize(size, seed=42):
    np.random.seed(seed)
    center = np.random.normal(0, 2, size=(4, 8))

    array = np.zeros((size, 14))

    tmp = np.zeros_like(array[:, :8])
    for i in range(size):
        cluster = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])
        tmp[i] = center[cluster - 1] + np.random.normal(0, cluster / 10, size=8)
    array[:, :8] = tmp

    center = np.random.normal(0, 2, size=(4, 4))
    tmp = np.zeros_like(array[:, 8:12])
    answers = []
    for i in range(size):
        cluster = np.random.choice([1, 2, 3, 4], p=[0.1, 0.2, 0.3, 0.4])
        tmp[i] = center[cluster - 1] + np.random.normal(0, cluster / 10, size=4)
        answers.append(cluster)
    array[:, 8:12] = tmp

    array[:, 12:] = np.random.normal(size=(size, 2))
    return array, answers
