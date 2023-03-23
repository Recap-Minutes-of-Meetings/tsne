import numpy as np


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
