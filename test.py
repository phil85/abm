import numpy as np
from abm import run_abm

# Set parameters
control_parameters = {}
control_parameters['output_as_pairs'] = True

# Illustrative example
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

pairs = run_abm(X, params=control_parameters)

for i, j in pairs:
    print(f'[{i}, {j}]')