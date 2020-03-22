
import numpy as np
import unittest
import matplotlib.pyplot as plt


def generate_gaussian_matrix(horiz_size, vert_size, mean: float = 0.0, standard_deviation: float = 1.0) -> np.array:
    return np.random.normal(loc=mean, scale=standard_deviation, size=(horiz_size, vert_size))


def generate_truncated_gaussian(horiz_size, vert_size, mean: float = 0.0, standard_deviation: float = 1.0,
                                min_size=-np.inf, max_size=np.inf) -> np.array:
    gaussian = generate_gaussian_matrix(horiz_size, vert_size, mean, standard_deviation)

    gaussian = np.maximum(gaussian, min_size, out=gaussian)
    gaussian = np.minimum(gaussian, max_size, out=gaussian)

    return gaussian


def generate_negative_truncated_gaussian(horiz_size, vert_size, mean: float = 0.0, standard_deviation: float = 1.0)\
        -> np.array:
    return generate_truncated_gaussian(horiz_size, vert_size, mean, standard_deviation, min_size=0.0)


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

    def test_truncated_gaussian_shape(self):
        horiz = 4
        vert = 8
        test_mat = np.zeros((4, 8))

        self.assertEqual(test_mat.shape, generate_truncated_gaussian(horiz, vert).shape)

    def test_truncated_gaussian_min(self):
        horiz = 400
        vert = 500

        required_min_val = -1.0
        min_val = np.min(generate_truncated_gaussian(horiz, vert, min_size=required_min_val))

        self.assertGreaterEqual(min_val, required_min_val)

    def test_truncated_gaussian_max(self):
        horiz = 400
        vert = 500

        required_max_val = -1.0
        max_val = np.max(generate_truncated_gaussian(horiz, vert, max_size=required_max_val))

        self.assertLessEqual(max_val, required_max_val)

    def test_truncated_gaussian_distribution(self):
        horiz = 100
        vert = 250

        res = np.zeros(horiz * vert)

        gaussian_matrix = generate_truncated_gaussian(horiz, vert)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

    def test_truncated_gaussian_distribution_truncate_min(self):
        horiz = 100
        vert = 250

        res = np.zeros(horiz * vert)

        gaussian_matrix = generate_truncated_gaussian(horiz, vert, min_size=-0.75)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

        gaussian_matrix = generate_truncated_gaussian(horiz, vert, min_size=-1.75)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

        gaussian_matrix = generate_truncated_gaussian(horiz, vert, min_size=-2.75)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

    def test_truncated_gaussian_distribution_truncate_max(self):
        horiz = 100
        vert = 250

        res = np.zeros(horiz * vert)

        gaussian_matrix = generate_truncated_gaussian(horiz, vert, max_size=-0.75)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

        gaussian_matrix = generate_truncated_gaussian(horiz, vert, max_size=0.25)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

        gaussian_matrix = generate_truncated_gaussian(horiz, vert, max_size=2.0)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()

    def test_negative_truncated_gaussian_distribution(self):
        horiz = 100
        vert = 250

        res = np.zeros(horiz * vert)

        gaussian_matrix = generate_negative_truncated_gaussian(horiz, vert)

        for i in range(horiz):
            for j in range(vert):
                res[i * vert + j] = gaussian_matrix[i, j]

        plt.hist(res, bins=100)
        plt.show()
