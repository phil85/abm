# © 2026, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import numpy as np


def run_random_matching(X, random_seed=24):

    # Get object ids
    object_ids = np.arange(X.shape[0])

    # Get number of anticlusters
    n_clusters = int(np.ceil(X.shape[0] / 2))

    # Set random seed
    np.random.seed(random_seed)

    # Randomly shuffle object ids
    np.random.shuffle(object_ids)

    # Assign objects to anticlusters
    labels = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        labels[object_ids[i]] = i % n_clusters

    return labels