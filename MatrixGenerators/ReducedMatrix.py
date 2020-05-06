import numpy as np
from typing import Tuple, List, Set


def reduce_block_matrix(block_matrix: np.ndarray, n: int, m: {int, np.ndarray}, lp_norm: float = 1.0) -> np.ndarray:
    reduced_matrix = np.zeros((n, n))
    if isinstance(m, int):
        m = np.ones((n,), dtype=int) * m
    for i in range(n):
        for j in range(n):
            submatrix_for_i_j = block_matrix[i * m[i]: (i + 1) * m[i], j * m[j]:(j + 1) * m[j]]
            reduced_matrix[i, j] = np.linalg.norm(submatrix_for_i_j.flatten(), lp_norm)
    return reduced_matrix


def coarse_element(matrix: np.ndarray, cluster_a: {int, np.ndarray, tuple}, cluster_b: {int, np.ndarray, tuple},
                   lp_norm: float) -> float:
    indexes = np.array(np.meshgrid(cluster_a, cluster_b)).T.reshape(-1, 2)
    values = np.asarray([matrix[tuple(index)] for index in indexes])
    return np.linalg.norm(values.flatten(), lp_norm)


def coarse_matrix(matrix: np.ndarray, clusters: {list, tuple}, lp_norm: float = 1.0) -> np.ndarray:
    n = len(clusters)
    coarsed_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if coarsed_matrix[j, i] != 0:
                coarsed_matrix[i, j] = coarsed_matrix[j, i]
            else:
                coarsed_matrix[i, j] = coarse_element(matrix, clusters[i], clusters[j], lp_norm)
    return coarsed_matrix





