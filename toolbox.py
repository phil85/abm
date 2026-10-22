import numpy as np

def compute_euclidean_distances(X, labels):

    # Get object ids in each anticluster 
    idx = np.argsort(labels).reshape(-1, 2)

    # Compute squared Euclidean distances 
    diff = X[idx[:, 0], :] - X[idx[:, 1], :]

    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def compute_squared_euclidean_distances(X, labels):

    # Get object ids in each anticluster 
    idx = np.argsort(labels).reshape(-1, 2)

    # Compute squared Euclidean distances 
    diff = X[idx[:, 0], :] - X[idx[:, 1], :]

    return np.einsum('ij,ij->i', diff, diff)


def compute_cityblock_distances(X, labels):

    # Get object ids in each anticluster 
    idx = np.argsort(labels).reshape(-1, 2)

    # Compute cityblock distances 
    diff = X[idx[:, 0], :] - X[idx[:, 1], :]

    return np.sum(np.abs(diff), axis=1)