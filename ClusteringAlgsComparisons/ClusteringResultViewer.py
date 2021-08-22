import pandas as pd
import numpy as np
import seaborn as sns
import random
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product


def get_random_colors_list(num_colors: int) -> list:
    colors_list = []
    for i in range(num_colors):
        r = random.random()
        b = random.random()
        g = random.random()
        colors_list.append((r, g, b))

    return colors_list


# TODO: change show to get so it'll by savable
def show_result_matrix(matrix, clusters, test_description=None):
    colors = get_random_colors_list(len(clusters))

    ax = sns.heatmap(matrix, annot=True, fmt='.1f', annot_kws={"fontsize": 8})

    for i, cluster in enumerate(clusters):
        if len(cluster) == 1:
            ax.add_patch(Rectangle((list(cluster)[0], list(cluster)[0]), 1, 1, fill=False, edgecolor=colors[i], lw=2))
        else:
            perm = product(list(cluster), repeat=2)
            for cell in list(perm):
                ax.add_patch(Rectangle(cell, 1, 1, fill=False, edgecolor=colors[i], lw=2))
    plt.title(test_description)
    plt.show()


class ClusteringResultViewer:
    __ALL_ALGORITHMS = ['Greedy',
                        'GreedyLoop',
                        'Random',
                        'Blossom',
                        'Annealing_Greedy',
                        'Hill_Greedy',
                        'RandomHill_Greedy']

    def __init__(self, results_df: pd.DataFrame, matrices_dir: str):
        self.results_df = results_df.copy()
        self.matrices_dir = matrices_dir

    def show_single_result_by_index(self, index: int):
        assert index < self.results_df.__len__()
        clusters = self.get_clusters_list(index)
        matrix = self.get_matrix(self.results_df.iloc[index]['matrix_id'])
        test_description = self.get_test_description(index)
        show_result_matrix(matrix, clusters, test_description)

    def show_single_result_by_matrix(self, matrix_id: str, algorithms: list = __ALL_ALGORITHMS):
        pass  # TODO

    def show_random_single_result_by(self,
                                     algorithms:               list  = __ALL_ALGORITHMS,
                                     agents_num:               list  = 24,
                                     cluster_size:             list   = 8,
                                     max_data_loss_percentage: float = 0.3,
                                     max_time_delta_seconds:   float = 1.0,
                                     max_average_cluster_size: float = 16.0):

        partial_conditions_df = pd.DataFrame(columns=list(self.results_df.columns))
        for i in range(len(algorithms)):
            tmp_df = self.results_df[self.results_df['algorithm'] == algorithms[i]]
            partial_conditions_df = pd.concat([partial_conditions_df, tmp_df])

        all_conditions_df = partial_conditions_df
        partial_conditions_df = pd.DataFrame()
        for i in range(len(agents_num)):
            tmp_df = all_conditions_df[all_conditions_df['agents_num'] == agents_num[i]]
            partial_conditions_df = pd.concat([partial_conditions_df, tmp_df])

        all_conditions_df = partial_conditions_df
        partial_conditions_df = pd.DataFrame()
        for i in range(len(cluster_size)):
            tmp_df = all_conditions_df[all_conditions_df['cluster_size'] == cluster_size[i]]
            partial_conditions_df = pd.concat([partial_conditions_df, tmp_df])

        all_conditions_df = partial_conditions_df
        all_conditions_df = all_conditions_df[all_conditions_df['dataLoss_percentage'] <= max_data_loss_percentage]
        all_conditions_df = all_conditions_df[all_conditions_df['time_delta'] <= max_time_delta_seconds]
        all_conditions_df = all_conditions_df[all_conditions_df['average_cluster_size'] <= max_average_cluster_size]

        index = all_conditions_df.sample(1).index[0]
        clusters = self.get_clusters_list(index)
        matrix = self.get_matrix(self.results_df.iloc[index]['matrix_id'])
        test_description = self.get_test_description(index)
        show_result_matrix(matrix, clusters, test_description)

    def show_random_n_results_by(self,
                                 n_results:                int   = 1,
                                 algorithms:               list  = __ALL_ALGORITHMS,
                                 num_agents:               list  = 24,
                                 cluster_size:             int   = 8,
                                 max_data_loss_percentage: float = 0.3,
                                 max_time_delta_seconds:   float = 1.0,
                                 max_average_cluster_size: float = 8.0):
        pass  # TODO

    def get_clusters_list(self, index: int) -> list:
        clusters_list = self.results_df.iloc[index]['output_clusters'].strip('][').split("}, {")
        clusters_sets_list = []
        for cluster in clusters_list:
            cluster_set = set(np.int_(cluster.strip('}{').split(', ')))
            clusters_sets_list.append(cluster_set)
        return clusters_sets_list

    def get_matrix(self, matrix_id: str) -> np.ndarray:
        matrix_file_name = matrix_id + '.csv'
        matrix_path = os.path.join(self.matrices_dir, matrix_file_name)
        file = open(matrix_path)
        matrix = np.loadtxt(file, delimiter=",")
        return matrix

    def get_test_description(self, index: int):
        test = self.results_df.iloc[index]

        algorithm            = test['algorithm']
        num_agents           = test['agents_num']
        cluster_size         = test['cluster_size']
        data_loss_percentage = test['dataLoss_percentage']
        time_delta           = test['time_delta']
        average_cluster_size = test['average_cluster_size']

        test_description = f'Tests Description: index={index}\n' \
                           f'Algorithm: {algorithm},   Run time: {time_delta:.3g}[s]\n' \
                           f'# Agents: {num_agents},   Cluster size: {cluster_size}\n' \
                           f'Data loss: {100*data_loss_percentage:.2g}%,   Average cluster size: {average_cluster_size}'
        return test_description

