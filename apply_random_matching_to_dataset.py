import time 
import numpy as np
import pandas as pd
from random_matching import run_random_matching
from toolbox import compute_squared_euclidean_distances, compute_cityblock_distances, compute_euclidean_distances

# Define metric
metric = 'sqeuclidean' # Options: 'sqeuclidean', 'euclidean', 'cityblock'

# Load dataset
X = pd.read_csv('datasets/npi.csv', header=None).to_numpy()

# Start stopwatch
tic = time.perf_counter()

# Run Random Matching
labels = run_random_matching(X, random_seed=1)

# Record running time
running_time = time.perf_counter() - tic

# Print running time
print(f'Running time [s] {running_time:.5f}')

# Compute total distance
if metric == 'sqeuclidean':
    total_distance = np.sum(compute_squared_euclidean_distances(X, labels))
elif metric == 'euclidean':
    total_distance = np.sum(compute_euclidean_distances(X, labels))
elif metric == 'cityblock':
    total_distance = np.sum(compute_cityblock_distances(X, labels))
print(f'Total distance {total_distance:.5f}')
