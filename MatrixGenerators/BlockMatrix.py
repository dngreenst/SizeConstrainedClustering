
import unittest
import numpy as np

from MatrixGenerators import TruncatedGaussian


def generate_block_negative_truncated_gaussian(n: int, m: int, mean: float = 0.0, standard_deviation: float = 1.0)\
        -> np.array:

    if m <= 0 or n <= 0:
        raise RuntimeError(f'Expected n, m > 0, received n={n}, m={m}')

    negative_truncated_gaussian = TruncatedGaussian.generate_negative_truncated_gaussian(n*m, n*m, mean,
                                                                                         standard_deviation)

    negative_truncated_gaussian = 0.5 * (negative_truncated_gaussian + negative_truncated_gaussian.T)

    for i in range(n):
        for j in range(m):
            for k in range(m):
                if k == j:
                    continue

                negative_truncated_gaussian[i*m + j, i*m + k] = 0.0

    return negative_truncated_gaussian


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

                    self.assertEqual(block_matrix[i*m + j, i*m + k], 0.0)

    def test_generate_gaussian_validate_symmetry(self):

        n = 40
        m = 80

        block_matrix = generate_block_negative_truncated_gaussian(n, m)

        self.assertTrue(np.array_equal(block_matrix, block_matrix.T))
        self.assertEqual(np.min(block_matrix - block_matrix.T), 0.0)
        self.assertEqual(np.max(block_matrix - block_matrix.T), 0.0)
