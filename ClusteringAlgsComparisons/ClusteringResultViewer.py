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


def show_2_results_of_matrix(matrix, clusters_a, clusters_b, test_description_a=None, test_description_b=None):
    colors = get_random_colors_list(max(len(clusters_a), len(clusters_b)))
    fig, (ax1, ax2) = plt.subplots(1, 2)

    sns.heatmap(matrix, cbar=False, ax=ax1)
    for i, cluster in enumerate(clusters_a):
        if len(cluster) == 1:
            ax1.add_patch(Rectangle((list(cluster)[0], list(cluster)[0]), 1, 1, fill=False, edgecolor=colors[i], lw=2))
        else:
            perm = product(list(cluster), repeat=2)
            for cell in list(perm):
                ax1.add_patch(Rectangle(cell, 1, 1, fill=False, edgecolor=colors[i], lw=2))
    ax1.set_title(test_description_a)

    sns.heatmap(matrix, cbar=False, ax=ax2)
    for i, cluster in enumerate(clusters_b):
        if len(cluster) == 1:
            ax2.add_patch(Rectangle((list(cluster)[0], list(cluster)[0]), 1, 1, fill=False, edgecolor=colors[i], lw=2))
        else:
            perm = product(list(cluster), repeat=2)
            for cell in list(perm):
                ax2.add_patch(Rectangle(cell, 1, 1, fill=False, edgecolor=colors[i], lw=2))
    ax2.set_title(test_description_b)

    plt.show()


def get_diff(row, field, df):
    diff = row[field] - df[df['matrix_id'] == row['matrix_id']][field].values[0]
    return diff


class ClusteringResultViewer:
    __ALL_ALGORITHMS = ['Greedy',
                        'GreedyLoop',
                        'Random',
                        'MaxWeight',
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
                                     min_data_loss_percentage: float = 0,
                                     max_time_delta_seconds:   float = 1.0,
                                     min_time_delta_seconds:   float = 0.0,
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
        all_conditions_df = all_conditions_df[(min_data_loss_percentage <= all_conditions_df['dataLoss_percentage']) &
                                              (all_conditions_df['dataLoss_percentage'] <= max_data_loss_percentage)]

        all_conditions_df = all_conditions_df[(min_time_delta_seconds <= all_conditions_df['time_delta']) &
                                              (all_conditions_df['time_delta'] <= max_time_delta_seconds)]

        all_conditions_df = all_conditions_df[all_conditions_df['average_cluster_size'] <= max_average_cluster_size]

        if all_conditions_df.empty:
            print('No results for set conditions')
            exit(0)

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

    def show_result_by_difference(self, algo_a, algo_b, min_diff, max_diff, field):
        # show test results of algo_a and algo_b where the difference in field was between min_diff to max_diff
        # difference = algo_a.field - algo_b.field

        df = self.results_df.copy()
        df_algo_a = df[df['algorithm'] == algo_a]
        df_algo_b = df[df['algorithm'] == algo_b]

        df_algo_a['diff'] = df_algo_a.apply(lambda row: get_diff(row, field, df_algo_b), axis=1)
        df_algo_a = df_algo_a[(df_algo_a['diff'] >= min_diff) & (df_algo_a['diff'] <= max_diff)]
        algo_a_idx = df_algo_a.sample(1).index[0]
        algo_b_idx = df_algo_b[df_algo_b['matrix_id'] == df_algo_a.loc[algo_a_idx]['matrix_id']].index[0]
        matrix = self.get_matrix(self.results_df.iloc[df_algo_a.sample(1).index[0]]['matrix_id'])
        clusters_a = self.get_clusters_list(algo_a_idx)
        clusters_b = self.get_clusters_list(algo_b_idx)
        test_description_a = self.get_test_description(algo_a_idx)
        test_description_b = self.get_test_description(algo_b_idx)
        show_2_results_of_matrix(matrix, clusters_a, clusters_b, test_description_a, test_description_b)

