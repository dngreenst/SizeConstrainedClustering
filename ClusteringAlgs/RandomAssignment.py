
import unittest
import numpy as np

from typing import List, Set


def random_permutation_clustering(block_matrix: np.ndarray, n: int, m: int, max_cluster_size: int) -> List[Set[int]]:

    if n <= 0 or m <= 0:
        raise RuntimeError(f'Expected n, m > 0, received n={n}, m={m}')

    if len(block_matrix.shape) != 2 or block_matrix.shape[0] != block_matrix.shape[1] or block_matrix.shape[0] != n*m:
        raise RuntimeError(f'Expected a square matrix made of m*m blocks.\n'
                           f'block_matrix.shape={block_matrix.shape}\n'
                           f'n*m={n*m}\n')

    perm = np.random.permutation(np.arange(n))

    clusters = list()
    curr_cluster_size = 0
    curr_cluster = None

    for idx in perm:
        if curr_cluster_size % max_cluster_size == 0:
            curr_cluster = set()
            clusters.append(curr_cluster)
        curr_cluster.add(int(idx))
        curr_cluster_size += 1

    return clusters


def random_permutation_clustering_from_matrix(matrix: np.array, max_cluster_size: int) -> List[Set[int]]:
    return random_permutation_clustering(matrix, matrix.shape[0], 1, max_cluster_size)


class RandomPermutationClusteringTester(unittest.TestCase):

    def test_random_perm_clustering_shape(self):
        matrix = np.zeros((4, 4, 4))

        raised_exception = False
        try:
            random_permutation_clustering(matrix, 4, 5, 3)
        except RuntimeError as re:
            self.assertEqual(str(re), f'Expected a square matrix made of m*m blocks.\n'
                                      f'block_matrix.shape={matrix.shape}\n'
                                      f'n*m={4*5}\n')
            raised_exception = True

        self.assertTrue(raised_exception)

    def test_random_perm_clustering_not_square(self):
        matrix = np.zeros((2, 8))

        raised_exception = False
        try:
            random_permutation_clustering(matrix, 4, 4, 3)
        except RuntimeError as re:
            self.assertEqual(str(re), f'Expected a square matrix made of m*m blocks.\n'
                                      f'block_matrix.shape={matrix.shape}\n'
                                      f'n*m={4*5}\n')
            raised_exception = True

        self.assertTrue(raised_exception)

    def test_random_perm_clustering_mn_not_positive(self):
        matrix = np.zeros((2, 8))

        raised_exception = False
        try:
            random_permutation_clustering(matrix, 4, -4, 3)
        except RuntimeError as re:
            self.assertEqual(str(re), f'Expected n, m > 0, received n={4}, m={-4}')
            raised_exception = True

        self.assertTrue(raised_exception)

    def test_random_perm_clustering_not_block(self):
        matrix = np.zeros((5, 5))

        raised_exception = False
        try:
            random_permutation_clustering(matrix, 4, 4, 3)
        except RuntimeError as re:
            self.assertEqual(str(re), f'Expected a square matrix made of m*m blocks.\n'
                                      f'block_matrix.shape={matrix.shape}\n'
                                      f'n*m={4*4}\n')
            raised_exception = True

        self.assertTrue(raised_exception)

    def test_random_perm_clustering_validate_cluster_size_limit(self):

        n = 40
        m = 80
        cluster_size_limit = 15

        block_matrix = np.zeros((n*m, n*m))
        clusters = random_permutation_clustering(block_matrix, n, m, cluster_size_limit)

        for cluster in clusters:
            self.assertLessEqual(len(cluster), cluster_size_limit)
