import time 
import numpy as np
from abm import run_abm
from toolbox import compute_squared_euclidean_distances, compute_cityblock_distances, compute_euclidean_distances

# Define data of the illustrative example
X = np.array([
    [7, 5], # 0
    [5, 4], # 1
    [7, 8], # 2
    [8, 5], # 3
    [4, 9], # 4 
    [5, 2], # 5
    [5, 10], # 6
    [2, 2], # 7
    [3, 1], # 8
    [1, 8], # 9
    ])

# Set parameters
control_parameters = {}
control_parameters['use_improvement'] = True
control_parameters['n_improvement_iterations'] = 2 
control_parameters['random_seed'] = 24 

# Start stopwatch
tic = time.perf_counter()

# Run abm
labels = run_abm(X, params=control_parameters)

# Record running time
running_time = time.perf_counter() - tic

# Print running time
print(f'Running time [s] {running_time:.5f}')

# Compute total distance
metric = control_parameters.get('metric', 'sqeuclidean')
if metric == 'sqeuclidean':
    total_distance = np.sum(compute_squared_euclidean_distances(X, labels))
elif metric == 'euclidean':
    total_distance = np.sum(compute_euclidean_distances(X, labels))
elif metric == 'cityblock':
    total_distance = np.sum(compute_cityblock_distances(X, labels))
print(f'Total distance {total_distance:.5f}')
