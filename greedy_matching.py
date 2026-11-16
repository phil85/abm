# © 2026, University of Bern, Group for Business Analytics, Operations Research and Quantitative Methods,
# Philipp Baumann

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform


def run_greedy_matching(X, metric):

    # Create a graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(X.shape[0]))

    # Compute distances
    try:
        distances = squareform(pdist(X, metric=metric))

        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                G.add_edge(i, j, weight=distances[i, j])

        # Find maximum weight matching
        matched = set()
        matching = set()
        for u, v, w in sorted(G.edges(data="weight"), key=lambda x: -x[2]):
            if u not in matched and v not in matched:
                matching.add((u, v))
                matched.add(u)
                matched.add(v)

    except MemoryError:
        print("Run out of memory error!")
        return np.array([]), None   

    # Get labels
    labels = np.zeros(X.shape[0], dtype=int)
    for i, (u, v) in enumerate(matching):
        labels[u] = labels[v] = i

    return labels
