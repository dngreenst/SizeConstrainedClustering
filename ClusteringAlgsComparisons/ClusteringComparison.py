from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
import time
from datetime import datetime
import math
import random

from common.enums.EMatrixType import EMatrixType
from MatrixGenerators import BlockMatrix, ReducedMatrix, ScatterBasedMatrix
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
        self.timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        self.res_df = pd.DataFrame(columns=["test_num",
                                            "matrix_id",
                                            "algorithm",
                                            "agents_num",
                                            "missions_num",
                                            "cluster_size",
                                            "dataLoss",
                                            "dataLoss_percentage",
                                            "average_cluster_size",
                                            "time_delta",
                                            "dataIn_percentage",
                                            "output_clusters"])

    def create_block_matrix(self) -> np.ndarray:
        # return BlockMatrix.generate_block_negative_truncated_gaussian(self.agents_num, self.missions_num,
        #                                                               standard_deviation=100)
        clusters_num = int(np.floor(self.agents_num / self.cluster_size))
        remainder = self.agents_num - clusters_num * self.cluster_size
        in_cluster_mean = random.choice([0, 1, 2])
        in_cluster_deviation = random.choice([50, 100, 200])
        outlier_num = random.choice([1, 2, 4]) * clusters_num
        outlier_mean = random.choice([0, 1, 2])
        outlier_deviation = random.choice([0.5, 2, 8]) * in_cluster_deviation
        return BlockMatrix.nonnegative_cluster_matrix_with_outliers(clusters_num=clusters_num,
                                                                    cluster_size=self.cluster_size,
                                                                    in_cluster_element_mean=in_cluster_mean,
                                                                    in_cluster_element_deviation=in_cluster_deviation,
                                                                    outliers_num=outlier_num,
                                                                    outlier_elements_mean=outlier_mean,
                                                                    outliers_element_deviation=outlier_deviation,
                                                                    remainder=remainder)

    def create_scatter_based_matrix(self) -> np.ndarray:
        r = random.randint(50, 150)

        # # for N-Dims scatter map
        # n = self.agents_num
        # r_effective = r / (n ** 0.85)

        n = 2
        r_effective = r
        map_shape = tuple(np.full(n, r_effective))
        return ScatterBasedMatrix.generate_scatter_based_matrix(
            agents_num                  = self.agents_num,
            map_size                    = map_shape,
            fractal_growth_probability  = np.random.uniform(0.3, 0.7),
            fractal_deviation           = np.random.uniform(0.5, 2),
            cost_function               = ScatterBasedMatrix.negative_exponential_distance)

    def reduce_matrix(self, block_matrix: np.ndarray, lp_norm: float = 1.0) -> np.ndarray:
        return ReducedMatrix.reduce_block_matrix(block_matrix, self.agents_num, self.missions_num, lp_norm)

    def __do_block_cluster(self, block_matrix: np.ndarray, alg_func) -> List[Set[int]]:
        return alg_func(block_matrix, self.agents_num, self.missions_num, self.cluster_size)

    def __do_cluster(self, matrix: np.ndarray, alg_func) -> List[Set[int]]:
        return alg_func(matrix, self.cluster_size)

    @staticmethod
    def __do_regret(matrix: np.ndarray, clusters: List[Set[int]]) -> float:
        return DataLossEstimator.calculate_data_loss(matrix, clusters)

    def compare(self, alg_dict: dict, matrices_ids: list = None, matrix_type: EMatrixType = EMatrixType.SCATTER):
        for key in alg_dict.keys():
            self.data[key] = Result(self.tests_num)

        for test_iter in np.arange(self.tests_num):
            print("Running test {0}".format(test_iter + 1))

            if matrix_type == EMatrixType.SCATTER:
                matrix = self.create_scatter_based_matrix() if self.matrix is None else self.matrix
            elif matrix_type == EMatrixType.BLOCK:
                matrix = self.create_block_matrix() if self.matrix is None else self.matrix
            else:
                matrix = self.create_block_matrix() if self.matrix is None else self.matrix

            edge_sum = np.sum(matrix)
            results_dir_path = path.join('output', 'results')
            save_path = path.join(results_dir_path, "matrices", f"{matrices_ids[test_iter]}.csv")
            np.savetxt(save_path, matrix, delimiter=",")
            for key in alg_dict.keys():
                print("test {0}, Alg: {1}".format(test_iter + 1, key))
                time_start = time.time()
                clusters = self.__do_cluster(matrix, alg_dict[key])
                time_delta = time.time() - time_start
                average_cluster_size = self.agents_num / len(clusters)
                regret = self.__do_regret(matrix, clusters)
                """setting data"""
                result: Result = self.data[key]
                result.dataLoss[test_iter] = regret
                result.dataLoss_percentage[test_iter] = regret / edge_sum
                result.average_cluster_size[test_iter] = average_cluster_size
                result.time_delta[test_iter] = time_delta
                result.dataIn_percentage[test_iter] = (edge_sum - regret) / edge_sum
                iter_res = pd.Series([test_iter+1,
                                      matrices_ids[test_iter],
                                      key,
                                      self.agents_num,
                                      self.missions_num,
                                      self.cluster_size,
                                      result.dataLoss[test_iter],
                                      result.dataLoss_percentage[test_iter],
                                      result.average_cluster_size[test_iter],
                                      result.time_delta[test_iter],
                                      result.dataIn_percentage[test_iter],
                                      clusters], index=self.res_df.columns)
                self.res_df = self.res_df.append(iter_res, ignore_index=True)
                print(f"cluster_len: {str([len(c) for c in clusters])}, "
                      f"fitness: {result.dataIn_percentage[test_iter] * 100:.2f}%, "
                      f"time: {result.time_delta[test_iter]:.2f}s")

    @staticmethod
    def __create_figure(name: str, num: int):
        plt.figure(num)
        ax = plt.subplot(111)
        plt.title(name)
        return ax

    def show_data(self):
        self.__scatter_plot()
        self.__box_plot()

    def __line_plot(self):
        line_style = ['-', '--', '-.', ':']
        for att_index, att_key in enumerate(Result().attributes.keys(), start=1):
            ax = self.__create_figure(att_key, att_index + 10 * self.id)
            for index, key in enumerate(self.data.keys()):
                style = line_style[math.floor(index / 10) % len(line_style)]
                ax.plot(self.data[key].attributes[att_key], label=key, linestyle=style)
            ax.legend()
        plt.show(block=False)

    def __scatter_plot(self):
        df = self.res_df.copy()
        df = df[df['algorithm'] != 'Random']
        df['color'] = df['algorithm'].astype("category")

        df.plot.scatter(x       = 'dataLoss_percentage',
                        y       = 'time_delta',
                        c       = 'color',
                        cmap    = "viridis",
                        s       = df['average_cluster_size']**2.5,
                        logy    = True,
                        alpha   = 0.5,
                        title   = "Clustering Comparison\n(size = avg cluster size)",
                        grid    = True,
                        figsize = (18, 10))

        save_path = path.join(path.join('output', 'results'), f'scatter_full_{self.tests_num}_tests_{self.timestamp}.pdf'.replace(':', ''))
        plt.savefig(save_path)
        plt.show(block=False)

    def __box_plot(self):
        cols = ['algorithm', 'dataLoss_percentage', 'dataIn_percentage', 'time_delta', 'average_cluster_size']
        df = self.res_df[cols].copy()
        df['algorithm'] = df['algorithm'].astype("category")

        fig, ax_new = plt.subplots(2, 2, figsize=(18, 10), sharey='none')
        df.boxplot(by="algorithm", ax=ax_new, layout=(2, 2), grid=False)
        fig.suptitle(f'Clustering Comparison\n'
                     f'Agents={self.agents_num}, Missions={self.missions_num}, Max cluster size={self.cluster_size}')
        save_path = path.join(path.join('output', 'results'), f'boxplot_full_{self.tests_num}_tests_{self.timestamp}.pdf'.replace(':', ''))
        plt.savefig(save_path)
        plt.show(block=False)
