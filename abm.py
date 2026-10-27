# © 2026, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed


def process_initialization_batch(X, batch, offset, params):
    n = len(batch) // 2
    result = offset + get_initial_matching_without_decomposition(X[batch, :], n, params)
    return batch, result


def process_batch(batch_A, batch_B, X, labels, params):
    
    # Get batch labels
    batch_labels = labels[batch_B]

    # Get metric
    metric = params.get('metric', 'sqeuclidean')

    # Compute distances
    distances = cdist(X[batch_A, :], X[batch_B, :], metric)

    # Run assignment algorithm
    row_ind, col_ind = linear_sum_assignment(distances, maximize=True)

    return batch_A[row_ind], batch_labels[col_ind]


def get_initial_matching_without_decomposition(X, n_clusters, params):

    # Initialize labels
    labels = np.full(X.shape[0], -1)

    # Get metric
    metric = params.get('metric', 'sqeuclidean')

    # Compute distances to global centroid
    global_centroid = X.mean(axis=0)
    distances = cdist(X, [global_centroid], metric)

    # Sort objects by distance from global centroid
    sorted_objects = np.argsort(distances[:, 0])

    # Get list_A (all odd indexed objects) and list_B (all even indexed objects)
    list_A = sorted_objects[::2]
    list_B = sorted_objects[1::2]

    # Assign first objects to centers
    labels[list_A] = np.arange(n_clusters)

    # Compute distances between objects in list_A and list_B
    distances = cdist(X[list_B, :], X[list_A, :], metric)

    # Solve assignment problem to maximize total distance
    _, col_ind = linear_sum_assignment(distances, maximize=True)

    # Update labels
    labels[list_B] = col_ind

    return labels


def get_initial_matching_with_decomposition(X, params):

    # Get control parameters
    assignment_problem_size = 2 * params.get('assignment_problem_size_for_decomposition', 1000)
    n_jobs_decomposition = params.get('n_jobs_decomposition', None)
    random_seed = params.get('random_seed', 24)

    # Set random seed
    np.random.seed(random_seed)

    # Get object ids
    object_ids = np.arange(X.shape[0])

    # Randomly shuffle object ids
    np.random.shuffle(object_ids)

    # Split object ids into batches of size assignment_problem_size
    batches = np.array_split(object_ids, range(assignment_problem_size, len(object_ids), assignment_problem_size))

    # Initialize labels
    labels = np.full(X.shape[0], -1)

    if n_jobs_decomposition is not None and n_jobs_decomposition != 1:

        # Precompute offsets 
        batch_sizes = [len(batch) // 2 for batch in batches]
        offsets = np.cumsum([0] + batch_sizes[:-1])

        results = Parallel(n_jobs=n_jobs_decomposition, backend="threading")(
            delayed(process_initialization_batch)(X, batch, offset, params)
            for batch, offset in zip(batches, offsets)
        )

        for batch, result in results:
            labels[batch] = result

    else:
        # Initialize counter
        counter = 0

        for batch in batches:
            labels[batch] = counter + get_initial_matching_without_decomposition(X[batch, :], len(batch) // 2, params)
            counter += len(batch) // 2

    return labels


def get_initial_matching(X, n_pairs, params):

    use_decomposition = params.get('use_decomposition', False)

    if use_decomposition:
        labels = get_initial_matching_with_decomposition(X, params)
    else: 
        labels = get_initial_matching_without_decomposition(X, n_pairs, params)

    return labels


def improve_matching(X, n_objects, n_pairs, labels, params):

    # Get control parameters
    random_seed = params.get('random_seed', 24)
    assignment_problem_size = params.get('assignment_problem_size_for_improvement', 1000)
    n_jobs_improvement = params.get('n_jobs_improvement', None)
    n_iterations = params.get('n_improvement_iterations', 10)
    metric = params.get('metric', 'sqeuclidean')

    # Adjust assignment_problem_size 
    if assignment_problem_size == -1:
        assignment_problem_size = n_pairs

    # Get object ids
    object_ids = np.arange(n_objects)

    # Create random number generator
    rng = np.random.default_rng(random_seed)

    for _ in range(n_iterations):

        # Get object ids in each pair
        object_ids_in_pairs = object_ids[np.argsort(labels)].reshape(-1, 2)

        # Randomly permute the order of the pairs so that batches are composed differently in each iteration
        if assignment_problem_size < n_pairs: 
            object_ids_in_pairs = object_ids_in_pairs[rng.permutation(n_pairs)]

        # Randomly select one object from each pair and assign it to set A and the other to set B
        mask = rng.integers(0, 2, size=n_pairs, dtype=np.uint8).astype(bool)
        list_A = np.where(mask, object_ids_in_pairs[:, 0], object_ids_in_pairs[:, 1])
        list_B = np.where(mask, object_ids_in_pairs[:, 1], object_ids_in_pairs[:, 0])

        # Create batches of objects to process
        batches_A = np.array_split(list_A, range(assignment_problem_size, len(list_A), assignment_problem_size))
        batches_B = np.array_split(list_B, range(assignment_problem_size, len(list_B), assignment_problem_size))

        if n_jobs_improvement is not None and n_jobs_improvement != 1:

            # Process batches in parallel
            results = Parallel(n_jobs=n_jobs_improvement, backend="threading")(
                    delayed(process_batch)(batch_u, batch_v, X, labels, params)
                    for batch_u, batch_v in zip(batches_A, batches_B)
                )

            # Apply all updates after computing (avoids shared-memory write races)
            for idx, new_labels in results:
                labels[idx] = new_labels

        else:
            for _, (batch_A, batch_B) in enumerate(zip(batches_A, batches_B)):

                # Get batch labels
                batch_labels = labels[batch_B]

                # Compute distances
                distances = cdist(X[batch_A, :], X[batch_B, :], metric)

                # Run assignment algorithm
                row_ind, col_ind = linear_sum_assignment(distances, maximize=True)

                # Update labels
                labels[batch_A[row_ind]] = batch_labels[col_ind]  

    return labels

def run_abm(X, params):

    # Get number of objects
    n_objects = X.shape[0]

    # Get number of pairs
    n_pairs = int(np.ceil(n_objects / 2))

    # Get initial matching
    labels = get_initial_matching(X, n_pairs, params)

    # Improve initial matching
    if params.get('use_improvement', False):
        labels = improve_matching(X, n_objects, n_pairs, labels, params)

    # Convert labels to set of pairs if desired
    if params.get('output_as_pairs', False):

        # The conversion of labels to pairs can be done in O(n) with a loop. For small n sorting turns out to be faster.
        order = np.argsort(labels, kind="stable")  
        pairs = order.reshape(-1, 2)               
        pair_set = {tuple(p) for p in pairs}
        return pair_set
    else:
        return labels