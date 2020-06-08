import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
import time
import math

from MatrixGenerators import BlockMatrix, ReducedMatrix
from RegretEstimators import DataLossEstimator

class Result:
    def __init__(self, tests_num: int = 0):
        self.dataLoss = np.zeros(tests_num)
        self.dataLoss_percentage = np.zeros(tests_num)
        self.average_cluster_size = np.zeros(tests_num)
        self.time_delta = np.zeros(tests_num)
        self.dataIn_percentage = np.zeros(tests_num)

        self.attributes = dict()
        # self.attributes["Data Loss"] = self.dataLoss
        # self.attributes["Data Loss Percentage"] = self.dataLoss_percentage
        self.attributes["Average Cluster Size"] = self.average_cluster_size
        self.attributes["Time Delta"] = self.time_delta
        self.attributes["Data In Cluster Percentage"] = self.dataIn_percentage


class ClusteringComparator:
    def __init__(self, agents_num: int, missions_num: int, cluster_size: int,
                 tests_num: int = 1, identifier: int = 0, block_matrix: np.ndarray = None):
        self.id = identifier
        self.tests_num = tests_num
        self.agents_num = agents_num
        self.missions_num = missions_num
        self.cluster_size = cluster_size
        self.data = dict()
        self.matrix = block_matrix

    def create_block_matrix(self) -> np.ndarray:
        return BlockMatrix.generate_block_negative_truncated_gaussian(self.agents_num, self.missions_num,
                                                                      standard_deviation=100)

    def reduce_matrix(self, block_matrix: np.ndarray, lp_norm: float = 1.0) -> np.ndarray:
        return ReducedMatrix.reduce_block_matrix(block_matrix, self.agents_num, self.missions_num, lp_norm)

    def __do_block_cluster(self, block_matrix: np.ndarray, alg_func) -> List[Set[int]]:
        return alg_func(block_matrix, self.agents_num, self.missions_num, self.cluster_size)

    def __do_cluster(self, matrix: np.ndarray, alg_func) -> List[Set[int]]:
        return alg_func(matrix, self.cluster_size)

    @staticmethod
    def __do_regret(matrix: np.ndarray, clusters: List[Set[int]]) -> float:
        return DataLossEstimator.calculate_data_loss(matrix, clusters)

    def compare(self, alg_dict: dict) -> dict:
        for key in alg_dict.keys():
            self.data[key] = Result(self.tests_num)

        for test_iter in np.arange(self.tests_num):
            print("Running test {0}".format(test_iter + 1))
            block_matrix = self.create_block_matrix() if self.matrix is None else self.matrix
            reduced_matrix = self.reduce_matrix(block_matrix)
            reduced_edge_sum = np.sum(reduced_matrix)
            for key in alg_dict.keys():
                print("test {0}, Alg: {1}".format(test_iter + 1, key))
                time_start = time.time()
                clusters = self.__do_cluster(reduced_matrix, alg_dict[key])
                time_delta = time.time() - time_start
                average_cluster_size = self.agents_num / len(clusters)
                regret = self.__do_regret(reduced_matrix, clusters)
                """setting data"""
                result: Result = self.data[key]
                result.dataLoss[test_iter] = regret
                result.dataLoss_percentage[test_iter] = regret / reduced_edge_sum
                result.average_cluster_size[test_iter] = average_cluster_size
                result.time_delta[test_iter] = time_delta
                result.dataIn_percentage[test_iter] = (reduced_edge_sum - regret) / reduced_edge_sum
                print("cluster_len: {0}, fitness: {1}".format(str([len(c) for c in clusters]),
                                                              result.dataIn_percentage[test_iter]))

    @staticmethod
    def __create_figure(name: str, num: int):
        plt.figure(num)
        ax = plt.subplot(111)
        plt.title(name)
        return ax

    def show_data(self):
        line_style = ['-', '--', '-.', ':']
        for att_index, att_key in enumerate(Result().attributes.keys(), start=1):
            ax = self.__create_figure(att_key, att_index + 10*self.id)
            for index, key in enumerate(self.data.keys()):
                style = line_style[math.floor(index / 10) % len(line_style)]
                ax.plot(self.data[key].attributes[att_key], label=key, linestyle=style)
            ax.legend()
        plt.show(block=False)
