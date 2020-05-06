
import unittest

import numpy as np
from typing import List, Set
from MatrixGenerators import ReducedMatrix


def calculate_data_loss(matrix: np.array, clusters: List[Set[int]], lp_norm: float = 1.0) -> float:

    # Input validation

    # Validate that the np.array is a matrix, and that it is square
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError(f'Only square matrices are supported')

    # Validate that:
    # 1. Each cluster contains only integers
    # 2. Each cluster contains distinct elements
    # 3. The clusters are pairwise disjoint
    # 4. The union of the clusters covers the matrix row indices
    matrix_dimension = matrix.shape[0]
    cluster_union = set.union(*clusters)
    sum_of_cluster_sizes = sum([len(cluster) for cluster in clusters])
    for cluster_member in cluster_union:
        if not isinstance(cluster_member, (int, np.integer)):
            raise RuntimeError(f'Unexpected cluster member type - {cluster_member}, {type(cluster_member)}')
    if len(cluster_union) != sum_of_cluster_sizes:
        raise RuntimeError(f'Invalid clusters - the clusters intersect')
    if sum_of_cluster_sizes != matrix_dimension:
        raise RuntimeError(f'Clusters don\'t match matrix dimensions: size of cluster\'s cluster_union is '
                           f'{sum_of_cluster_sizes}, matrix dimension is {matrix_dimension}')

    min_cluster_idx = min(cluster_union)
    max_cluster_idx = max(cluster_union)

    if min_cluster_idx != 0 or max_cluster_idx != matrix_dimension - 1:
        raise RuntimeError(f'cluster indices don\'t match matrix indices')

    # For each cluster C, create an indicator matrix, such that indicator[i, j] = 1 <==> i,j are C
    # Combine the indicator matrices to create a cumulative indicator matrix, such that
    # cumulative_indicator[i, j] = 1 <==> there is a cluster C in the provided clusters, such that i,j are in C
    clusters = [list(cluster) for cluster in clusters]
    cumulative_indicator = np.zeros_like(matrix)
    for cluster in clusters:
        indicator_array = np.zeros(matrix_dimension)
        # Assigning 1 at indices determined by a collection is available for lists, but not for numpy arrays
        indicator_array[cluster] = 1
        indicator_matrix = np.outer(indicator_array, indicator_array)
        cumulative_indicator += indicator_matrix

    cumulative_indicator_complement = np.ones_like(cumulative_indicator) - cumulative_indicator

    lost_data = np.multiply(cumulative_indicator_complement, matrix).reshape(matrix_dimension ** 2)

    # python complains, but a float argument for 'norm' acts as expected - that is, L_p norm.
    data_loss_size = np.linalg.norm(lost_data, lp_norm)

    return data_loss_size


def block_matrix_calculate_data_loss(block_matrix: np.array, n: int, m: int, clusters: List[Set[int]],
                                     lp_norm: float = 1.0) -> float:
    if n <= 0 or m <= 0:
        raise RuntimeError(f'Expected n, m > 0, received n={n}, m={m}')

    if len(block_matrix.shape) != 2 or block_matrix.shape[0] != block_matrix.shape[1] or block_matrix.shape[0] != n*m:
        raise RuntimeError(f'Expected a square matrix made of m*m blocks.\n'
                           f'block_matrix.shape={block_matrix.shape}\n'
                           f'n*m={n*m}\n')
    return calculate_data_loss(ReducedMatrix.reduce_block_matrix(block_matrix, n, m, lp_norm), clusters, 1.0)


class TestBlockMatrixCalculateDataLoss(unittest.TestCase):

    def test_calculate_data_loss_invalid_matrix_shape(self):
        block_matrix = np.arange(16 ** 2).reshape(16, 16)

        block_matrix_calculate_data_loss(block_matrix, 8, 2, [[1, 3, 5], [2, 4, 6], [7, 0]])


class TestCalculateDataLoss(unittest.TestCase):

    def test_calculate_data_loss_invalid_matrix_shape(self):
        try:
            calculate_data_loss(np.zeros((1, 2, 3)), list())
        except RuntimeError as re:
            self.assertEqual(str(re), "Only square matrices are supported")

    def test_calculate_data_loss_invalid_matrix_nonsquare(self):
        try:
            calculate_data_loss(np.zeros((2, 3)), list())
        except RuntimeError as re:
            self.assertEqual(str(re), "Only square matrices are supported")

    def test_calculate_data_loss_invalid_clusters_intersecting_clusters(self):

        clusters = list()

        cluster1 = [0, 1]
        cluster2 = [1, 2]

        clusters.append(cluster1)
        clusters.append(cluster2)

        try:
            calculate_data_loss(np.zeros((3, 3)), clusters)
        except RuntimeError as re:
            self.assertEqual(str(re), "Invalid clusters - the clusters intersect")

    def test_calculate_data_loss_invalid_clusters_non_integer_clusters(self):

        clusters = list()

        cluster1 = [0, 1]
        cluster2 = [1, 'Oh noes']

        clusters.append(cluster1)
        clusters.append(cluster2)

        try:
            calculate_data_loss(np.zeros((3, 3)), clusters)
        except RuntimeError as re:
            self.assertEqual(str(re), "Unexpected cluster member type - Oh noes, <class 'str'>")

    def test_calculate_data_loss_invalid_clusters_non_distinct_cluster(self):

        clusters = list()

        cluster1 = [0, 1]
        cluster2 = [2, 2]

        clusters.append(cluster1)
        clusters.append(cluster2)

        try:
            calculate_data_loss(np.zeros((3, 3)), clusters)
        except RuntimeError as re:
            self.assertEqual(str(re), "Invalid clusters - the clusters intersect")

    def test_calculate_data_loss_invalid_clusters_union_too_big(self):

        clusters = list()

        cluster1 = [0, 1, 5]
        cluster2 = [2, 4]

        clusters.append(cluster1)
        clusters.append(cluster2)

        try:
            calculate_data_loss(np.zeros((3, 3)), clusters)
        except RuntimeError as re:
            self.assertEqual(str(re),
                             "Clusters don\'t match matrix dimensions: size of cluster\'s cluster_union is 5, matrix dimension is 3")

    def test_calculate_data_loss_invalid_clusters_members_dont_cover_matrix_row_indices(self):

        clusters = list()

        cluster1 = [0, 1]
        cluster2 = [7]

        clusters.append(cluster1)
        clusters.append(cluster2)

        try:
            calculate_data_loss(np.zeros((3, 3)), clusters)
        except RuntimeError as re:
            self.assertEqual(str(re),
                             "cluster indices don\'t match matrix indices")

    def test_calculate_data_loss_trivial_partition(self):

        clusters = list()

        cluster1 = [0]
        cluster2 = [1]

        clusters.append(cluster1)
        clusters.append(cluster2)

        expected_data_loss = 2.0

        try:
            data_loss_res = calculate_data_loss(np.ones((2, 2)), clusters)
        except Exception as e:
            self.fail(f'Caught unexpected exception:\n\n{e}')

        self.assertEqual(data_loss_res, expected_data_loss)

    def test_calculate_data_loss_non_uniform_matrix_trivial_partition(self):

        clusters = list()

        cluster1 = [0]
        cluster2 = [1]

        clusters.append(cluster1)
        clusters.append(cluster2)

        data_matrix = np.array([[1.0, 2.0],
                                [5.5, 3.0]])
        expected_data_loss = 7.5

        try:
            data_loss_res = calculate_data_loss(data_matrix, clusters)
        except Exception as e:
            self.fail(f'Caught unexpected exception:\n\n{e}')

        self.assertEqual(data_loss_res, expected_data_loss)

    def test_calculate_data_loss_uniform_matrix_nontrivial_partition(self):

        clusters = list()

        cluster1 = [0, 4, 3, 6]
        cluster2 = [1]
        cluster3 = [9, 2, 5]
        cluster4 = [8, 7]

        clusters.append(cluster1)
        clusters.append(cluster2)
        clusters.append(cluster3)
        clusters.append(cluster4)

        data_matrix = np.ones((10, 10))
        expected_data_loss = 70.0

        try:
            data_loss_res = calculate_data_loss(data_matrix, clusters)
        except Exception as e:
            self.fail(f'Caught unexpected exception:\n\n{e}')

        self.assertEqual(data_loss_res, expected_data_loss)

    def test_calculate_data_loss_nonuniform_matrix_nontrivial_partition(self):

        clusters = list()

        cluster1 = [0, 4]
        cluster2 = [1, 3]
        cluster3 = [2]

        clusters.append(cluster1)
        clusters.append(cluster2)
        clusters.append(cluster3)

        data_matrix = np.array([
            [5.5, 16.6, 2.5, -8.3, 1.5],
            [2.45, 3.2, 2.1, 3.0, 13.3],
            [15.5, 17.42, 0.5, 11.1, 1.1],
            [4.44, 5.32, 1.11, 23.0, 4.1],
            [7.25, 9.12, 6.06, 19.91, 1.4]
        ])
        sum_of_data = np.sum(np.abs(data_matrix))
        in_cluster_data = 0.0
        for cluster in clusters:
            for first_cluster_element in cluster:
                for second_cluster_element in cluster:
                    in_cluster_data += np.abs(data_matrix[first_cluster_element, second_cluster_element])
        expected_data_loss = sum_of_data - in_cluster_data

        try:
            data_loss_res = calculate_data_loss(data_matrix, clusters)
        except Exception as e:
            self.fail(f'Caught unexpected exception:\n\n{e}')

        self.assertEqual(data_loss_res, expected_data_loss)

    def test_calculate_data_loss_uniform_matrix_nontrivial_norm(self):

        clusters = list()

        cluster1 = [0]
        cluster2 = [1]
        cluster3 = [2]

        clusters.append(cluster1)
        clusters.append(cluster2)
        clusters.append(cluster3)

        data_matrix = np.ones((3, 3))

        expected_data_loss = np.sqrt(6.0)

        try:
            data_loss_res = calculate_data_loss(data_matrix, clusters, lp_norm=2.0)
        except Exception as e:
            self.fail(f'Caught unexpected exception:\n\n{e}')

        self.assertAlmostEqual(data_loss_res, expected_data_loss)
