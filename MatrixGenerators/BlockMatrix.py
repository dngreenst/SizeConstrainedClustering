import unittest
from typing import Tuple

import numpy as np

from MatrixGenerators import TruncatedGaussian


def generate_block_negative_truncated_gaussian(n: int, m: int, mean: float = 0.0, standard_deviation: float = 1.0) \
        -> np.array:
    if m <= 0 or n <= 0:
        raise RuntimeError(f'Expected n, m > 0, received n={n}, m={m}')

    negative_truncated_gaussian = TruncatedGaussian.generate_negative_truncated_gaussian(n * m, n * m, mean,
                                                                                         standard_deviation)

    negative_truncated_gaussian = 0.5 * (negative_truncated_gaussian + negative_truncated_gaussian.T)

    for i in range(n):
        for j in range(m):
            for k in range(m):
                if k == j:
                    continue

                negative_truncated_gaussian[i * m + j, i * m + k] = 0.0

    return negative_truncated_gaussian


def nonnegative_cluster_matrix_with_outliers(clusters_num: int,
                                             cluster_size: int,
                                             in_cluster_element_mean: float,
                                             in_cluster_element_deviation: float,
                                             outliers_num: int,
                                             outlier_elements_mean: float,
                                             outliers_element_deviation: float,
                                             remainder: int) -> np.array:
    cluster_matrix = generate_cluster_matrix_with_outliers(clusters_num=clusters_num, cluster_size=cluster_size,
                                                           in_cluster_element_mean=in_cluster_element_mean,
                                                           in_cluster_element_deviation=in_cluster_element_deviation,
                                                           outliers_num=outliers_num,
                                                           outlier_elements_mean=outlier_elements_mean,
                                                           outliers_element_deviation=outliers_element_deviation,
                                                           remainder=remainder)

    np.maximum(cluster_matrix, 0.0, out=cluster_matrix)

    return cluster_matrix


def generate_cluster_matrix_with_outliers(clusters_num: int,
                                          cluster_size: int,
                                          in_cluster_element_mean: float,
                                          in_cluster_element_deviation: float,
                                          outliers_num: int,
                                          outlier_elements_mean: float,
                                          outliers_element_deviation: float,
                                          remainder: int) -> np.array:
    matrix_row_dim = (clusters_num * cluster_size)  + remainder

    result_matrix = _generate_symmetric_gaussian_block_matrix(mean=in_cluster_element_mean,
                                                              deviation=in_cluster_element_deviation,
                                                              clusters_num=clusters_num,
                                                              cluster_size=cluster_size,
                                                              matrix_row_dim=matrix_row_dim,
                                                              remainder=remainder)

    _add_outliers_to_block_matrix(mean=outlier_elements_mean,
                                  deviation=outliers_element_deviation,
                                  cluster_size=cluster_size,
                                  clusters_num=clusters_num,
                                  outliers_num=outliers_num,
                                  output_matrix=result_matrix,
                                  remainder=remainder)

    np.fill_diagonal(result_matrix, 0)

    return result_matrix  # TODO: revert to _permute_symmetrically(result_matrix)


def _permute_symmetrically(matrix: np.array) -> np.array:
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f'_permute_symmetrically requires a symmetric 2D array as input. '
                           f'Received an array of shape {matrix.shape}')
    row_dimension = matrix.shape[0]
    permutation = np.random.permutation(np.arange(row_dimension)).tolist()

    permuted_matrix = np.zeros_like(matrix)
    permuted_matrix[:, :] = matrix[permutation, :]
    permuted_matrix[:, :] = permuted_matrix[:, permutation]

    return permuted_matrix


def _generate_random_indices_in_different_clusters(cluster_size: int,
                                                   clusters_num: int,
                                                   remainder: int) -> Tuple[int, int]:
    dimension = (cluster_size * clusters_num) + remainder

    while True:
        i = np.random.randint(low=0, high=dimension)
        j = np.random.randint(low=0, high=dimension)

        if i % cluster_size != j % cluster_size:
            return i, j


def _add_outliers_to_block_matrix(mean: float,
                                  deviation: float,
                                  cluster_size: int,
                                  clusters_num: int,
                                  outliers_num: int,
                                  output_matrix: np.array,
                                  remainder: int) -> None:
    for _ in range(outliers_num):
        i, j = _generate_random_indices_in_different_clusters(cluster_size=cluster_size,
                                                              clusters_num=clusters_num,
                                                              remainder=remainder)

        outlier = np.random.normal(loc=mean,
                                   scale=deviation)
        output_matrix[i, j] += outlier
        output_matrix[j, i] = output_matrix[i, j]


def _generate_symmetric_gaussian_block_matrix(mean: float,
                                              deviation: float,
                                              clusters_num: int,
                                              cluster_size: int,
                                              matrix_row_dim: np.array,
                                              remainder: int) -> np.array:
    result_matrix = np.zeros((matrix_row_dim, matrix_row_dim))
    for cluster_index in range(clusters_num):
        # Generate blocks of symmetric gaussian matrices and insert them into the large matrix
        single_matrix_block = _symmetric_gaussian_matrix(mean=mean,
                                                         deviation=deviation,
                                                         block_size=cluster_size)
        min_block_index = cluster_index * cluster_size
        max_block_index = (cluster_index + 1) * cluster_size
        result_matrix[min_block_index: max_block_index, min_block_index: max_block_index] = single_matrix_block

    if remainder > 0:
        single_matrix_block = _symmetric_gaussian_matrix(mean=mean,
                                                         deviation=deviation,
                                                         block_size=remainder)
        min_block_index = clusters_num * cluster_size
        max_block_index = min_block_index + remainder
        result_matrix[min_block_index: max_block_index, min_block_index: max_block_index] = single_matrix_block

    return result_matrix


def _symmetric_gaussian_matrix(mean: float,
                               deviation: float,
                               block_size: int) -> np.array:
    # Generate a non-symmetric gaussian matrix
    gaussian_matrix = np.random.normal(loc=mean,
                                       scale=deviation,
                                       size=(block_size, block_size))

    # Make the matrix symmetric
    for i in range(block_size):
        for j in range(i):
            gaussian_matrix[j, i] = gaussian_matrix[i, j]

    return gaussian_matrix


class BlockMatrixGenerationTester(unittest.TestCase):

    def test_generate_gaussian_matrix_shape(self):

        n = 4
        m = -8
        raised_exception = False
        try:
            generate_block_negative_truncated_gaussian(n, m)
        except RuntimeError as re:
            self.assertEqual(str(re), "Expected n, m > 0, received n=4, m=-8")
            raised_exception = True

        self.assertTrue(raised_exception)

    def test_generate_gaussian_validate_zeroed_diag_blocks(self):

        n = 40
        m = 80

        block_matrix = generate_block_negative_truncated_gaussian(n, m)

        for i in range(n):
            for j in range(m):
                for k in range(m):
                    if j == k:
                        continue

                    self.assertEqual(block_matrix[i * m + j, i * m + k], 0.0)

    def test_generate_gaussian_validate_symmetry(self):

        n = 40
        m = 80

        block_matrix = generate_block_negative_truncated_gaussian(n, m)

        self.assertTrue(np.array_equal(block_matrix, block_matrix.T))
        self.assertEqual(np.min(block_matrix - block_matrix.T), 0.0)
        self.assertEqual(np.max(block_matrix - block_matrix.T), 0.0)

    def test_generate_cluster_matrix_with_outliers(self):

        clusters_num = 4
        cluster_size = 5

        in_cluster_element_mean = 0.5
        in_cluster_element_deviation = 1.0

        outliers_num = 3
        outliers_element_mean = 0.8
        outliers_element_deviation = 1.1

        block_matrix = generate_cluster_matrix_with_outliers(cluster_size=cluster_size,
                                                             clusters_num=clusters_num,
                                                             in_cluster_element_mean=in_cluster_element_mean,
                                                             in_cluster_element_deviation=in_cluster_element_deviation,
                                                             outliers_num=outliers_num,
                                                             outlier_elements_mean=outliers_element_mean,
                                                             outliers_element_deviation=outliers_element_deviation)

        print(block_matrix)

    def test_nonnegative_cluster_matrix_with_outliers(self):

        clusters_num = 3
        cluster_size = 2

        in_cluster_element_mean = 0.5
        in_cluster_element_deviation = 1.0

        outliers_num = 3
        outliers_element_mean = 0.8
        outliers_element_deviation = 1.1

        block_matrix = nonnegative_cluster_matrix_with_outliers(
            cluster_size=cluster_size,
            clusters_num=clusters_num,
            in_cluster_element_mean=in_cluster_element_mean,
            in_cluster_element_deviation=in_cluster_element_deviation,
            outliers_num=outliers_num,
            outlier_elements_mean=outliers_element_mean,
            outliers_element_deviation=outliers_element_deviation)

        self.assertTrue(np.all(block_matrix >= 0.0))
