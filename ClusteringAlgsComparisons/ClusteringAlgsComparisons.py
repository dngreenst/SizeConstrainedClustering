
import unittest
import numpy as np
import matplotlib.pyplot as plt

from MatrixGenerators import BlockMatrix
from RegretEstimators import DataLossEstimator
from ClusteringAlgs import RandomAssignment


class CompareClusteringAlgorithms(unittest.TestCase):

    def test_demonstrate_testing_method(self):

        tests_num = 50

        regrets = np.zeros(tests_num)
        regret_percentage = np.zeros(tests_num)

        n = 100
        m = 15
        max_cluster_size = 8

        for test_iter in range(tests_num):
            block_matrix = BlockMatrix.generate_block_negative_truncated_gaussian(n, m)

            clusters = RandomAssignment.random_permutation_clustering(block_matrix, n, m, max_cluster_size)

            regret = DataLossEstimator.block_matrix_calculate_data_loss(block_matrix, n, m, clusters)

            regrets[test_iter] = regret
            regret_percentage[test_iter] = regret / np.sum(np.abs(block_matrix))

        plt.plot(regrets)
        plt.show()
        plt.plot(regret_percentage)
        plt.show()
