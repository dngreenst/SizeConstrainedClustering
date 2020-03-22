
import numpy as np
import unittest
import matplotlib.pyplot as plt


def generate_gaussian_matrix(horiz_size, vert_size, mean: float = 0.0, standard_deviation: float = 1.0) -> np.array:
    return np.random.normal(loc=mean, scale=standard_deviation, size=(horiz_size, vert_size))


class MatrixGenerationTester(unittest.TestCase):

    def test_generate_gaussian_matrix_shape(self):
        horiz = 4
        vert = 8
        test_mat = np.zeros((4, 8))

        self.assertEqual(test_mat.shape, generate_gaussian_matrix(horiz, vert).shape)

    def test_generate_gaussian_matrix_distribution_standard_gaussian(self):

        horiz = 100
        vert = 250

        res = np.zeros(horiz * vert)

        gaussian_matrix = generate_gaussian_matrix(horiz, vert)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

    def test_generate_gaussian_matrix_custom_mean(self):
        horiz = 100
        vert = 250

        res = np.zeros(horiz * vert)

        gaussian_matrix = generate_gaussian_matrix(horiz, vert, mean=15)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

    def test_generate_gaussian_matrix_custom_standard_deviation(self):
        horiz = 100
        vert = 250

        res = np.zeros(horiz * vert)

        gaussian_matrix = generate_gaussian_matrix(horiz, vert, standard_deviation=2.0)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()
