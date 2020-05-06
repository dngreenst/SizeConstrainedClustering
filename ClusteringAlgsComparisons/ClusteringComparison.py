import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
import time

from MatrixGenerators import BlockMatrix, ReducedMatrix
from RegretEstimators import DataLossEstimator


class Result:
    def __init__(self, tests_num: int = 0):
        self.dataLoss = np.zeros(tests_num)
        self.dataLoss_percentage = np.zeros(tests_num)
        self.average_cluster_size = np.zeros(tests_num)
        self.time_delta = np.zeros(tests_num)

        self.attributes = dict()
        self.attributes["Data Loss"] = self.dataLoss
        self.attributes["Data Loss Percentage"] = self.dataLoss_percentage
        self.attributes["Average Cluster Size"] = self.average_cluster_size
        self.attributes["Time Delta"] = self.time_delta


class ClusteringComparator:
    def __init__(self, agents_num: int, missions_num: int, cluster_size: int, tests_num: int = 1):
        self.tests_num = tests_num
        self.agents_num = agents_num
        self.missions_num = missions_num
        self.cluster_size = cluster_size
        self.data = dict()

    def create_block_matrix(self) -> np.ndarray:
        return BlockMatrix.generate_block_negative_truncated_gaussian(self.agents_num, self.missions_num)

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

        for test_iter in range(self.tests_num):
            block_matrix = self.create_block_matrix()
            reduced_matrix = self.reduce_matrix(block_matrix)
            for key in alg_dict.keys():
                time_start = time.time()
                clusters = self.__do_cluster(reduced_matrix, alg_dict[key])
                time_delta = time.time() - time_start
                average_cluster_size = self.agents_num/len(clusters)
                regret = self.__do_regret(reduced_matrix, clusters)
                """setting data"""
                result: Result = self.data[key]
                result.dataLoss[test_iter] = regret
                result.dataLoss_percentage[test_iter] = regret / np.sum(np.abs(reduced_matrix))
                result.average_cluster_size[test_iter] = average_cluster_size
                result.time_delta[test_iter] = time_delta

    @staticmethod
    def __create_figure(name: str, num: int) -> plt.Subplot:
        plt.figure(num)
        ax = plt.subplot(111)
        plt.title(name)
        return ax

    def show_data(self):
        for index, att_key in enumerate(Result().attributes.keys()):
            ax = self.__create_figure(att_key, index+1)
            for key in self.data.keys():
                ax.plot(self.data[key].attributes[att_key], label=key)
            ax.legend()
        plt.show()