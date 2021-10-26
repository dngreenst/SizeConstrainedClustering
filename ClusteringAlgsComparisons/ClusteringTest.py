from ClusteringAlgsComparisons.ClusteringComparison import ClusteringComparator
from ClusteringAlgs import RandomAssignment
from ClusteringAlgs.HierarchyCluster import BlossomCluster
from ClusteringAlgs.HierarchyCluster import FidlerVecCluster
from ClusteringAlgs.HierarchyCluster import SpectralClustering
from ClusteringResultViewer import ClusteringResultViewer
import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt
from ClusteringAlgs.StochasticCluster import StochasticCluster
from datetime import datetime
import shutil
import os
from os import path
import uuid

import networkx as nx


class ClusteringTest:
    @staticmethod
    def multiple_parameters_testing_method(agents_num_list, cluster_size_list, missions, num_tests, algo_filter_list):
        total_num_tests = num_tests * len(agents_num_list) * len(cluster_size_list)
        results_dir_name = 'results'
        results_dir_path = os.path.join('output', results_dir_name)
        timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        text_identifier = f'{results_dir_name}_{num_tests}_tests_{timestamp}'
        matrices_dir_name = f'matrices_{num_tests}_tests_{timestamp}'
        matrices_dir_path = os.path.join('output', matrices_dir_name)

        os.makedirs(matrices_dir_path, exist_ok=True)
        if path.exists(results_dir_path):
            os.rename(results_dir_path, os.path.join('output', f'{results_dir_name}_prev'))

        total_res_df = pd.DataFrame()
        for agents in agents_num_list:
            for cluster_size in cluster_size_list:
                matrices_ids = [uuid.uuid1() for i in range(num_tests)]
                os.makedirs(results_dir_path, exist_ok=True)
                os.mkdir(os.path.join(results_dir_path, 'matrices'))
                curr_df = ClusteringTest.test_demonstrate_testing_method(agents           = agents,
                                                                         missions         = missions,
                                                                         cluster_size     = cluster_size,
                                                                         tests            = num_tests,
                                                                         plot_block       = False,
                                                                         algo_filter_list = algo_filter_list,
                                                                         matrices_ids     = matrices_ids)
                total_res_df = pd.concat([total_res_df, curr_df])

                # copy test matrices
                source_folder = os.path.join(results_dir_path, 'matrices')
                destination_folder = matrices_dir_path
                for file_name in os.listdir(source_folder):
                    source = os.path.join(source_folder, file_name)
                    destination = os.path.join(destination_folder, file_name)
                    if os.path.isfile(source):
                        shutil.copy(source, destination)

                os.rename(results_dir_path, os.path.join('output', f'{text_identifier}_{agents}_agents_{cluster_size}_cluster_size'))

        ClusteringTest.save_results_df(total_res_df, os.path.join('output', text_identifier))
        ClusteringTest.scatter_plot(total_res_df, total_num_tests, timestamp)
        ClusteringTest.box_plot(total_res_df, total_num_tests, timestamp)
        # plt.show()

    @staticmethod
    def test_demonstrate_testing_method(agents=24, missions=1, cluster_size=8, tests=50, plot_block=True,
                                        algo_filter_list=[], matrices_ids=None) -> pd.DataFrame():

        results_dir = 'results'
        results_dir_path = os.path.join('output', results_dir)
        matrices_dir = 'matrices'
        os.makedirs(path.join(results_dir_path, matrices_dir), exist_ok=True)

        if matrices_ids is None:
            matrices_ids = [i for i in range(tests)]

        comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size, tests_num=tests)
        alg_dict = dict()
        alg_dict["Random"] = RandomAssignment.random_permutation_clustering_from_matrix
        alg_dict["Blossom"] = BlossomCluster.BlossomCluster.cluster
        ClusteringTest.add_stochastic_algs(alg_dict)
        ClusteringTest.filer_algorithms_to_compare(algo_filter_list, alg_dict)
        comparator.compare(alg_dict, matrices_ids)
        ClusteringTest.save_results_df(comparator.res_df, path.join(results_dir_path,
                                                                    "results_{}".format(comparator.timestamp)))
        comparator.show_data()
        plt.show(block=plot_block)
        return comparator.res_df

    @staticmethod
    def add_fidler_algs(alg_dict, method_dict: dict, join_dict: dict, num_vectors: np.ndarray, num_clusters: np.ndarray):
        method_arr = list(method_dict.keys())
        join_arr = list(join_dict.keys())
        fidler_options = np.array(np.meshgrid(num_vectors, num_clusters, method_arr, join_arr)).T.reshape(-1, 4)
        for t in fidler_options:
            alg_dict["Fidler_{2}, vec: {0}, clust: {1}, join: {3}".format(t[0], t[1], t[2], t[3])] = \
                functools.partial(FidlerVecCluster.fidler_cluster, num_fidler_vectors=int(t[0]), num_clusters=int(t[1]),
                                  method_function=method_dict[t[2]], join_function=join_dict[t[3]])

    @staticmethod
    def add_spectral_algs(alg_dict: dict, join_dict: dict, num_vectors: np.ndarray, num_clusters: np.ndarray):
        join_arr = list(join_dict.keys())
        spectral_options = np.array(np.meshgrid(num_vectors, num_clusters, join_arr)).T.reshape(-1, 3)
        for t in spectral_options:
            alg_dict["Spectral, vec: {0}, clust: {1}, join: {2}".format(t[0], t[1], t[2])] = \
                functools.partial(SpectralClustering.cluster, num_fidler_vectors=int(t[0]), num_clusters=int(t[1]),
                                  join_function=join_dict[t[2]])

    @staticmethod
    def add_stochastic_algs(alg_dict: dict):
        alg_dict["Greedy"] = StochasticCluster.greedy_search
        alg_dict['GreedyLoop'] = functools.partial(StochasticCluster.greedy_loop,
                                                   solver=StochasticCluster.greedy_search, iter_num=100)
        for alg in ['Greedy']:
            alg_dict['Annealing_{0}'.format(alg)] = functools.partial(
                StochasticCluster.solve_with_initial_solver,
                init_solver=alg_dict[alg],
                solver_factory=StochasticCluster.annealing_solver_factory)
            alg_dict['Hill_{0}'.format(alg)] = functools.partial(
                StochasticCluster.solve_with_initial_solver,
                init_solver=alg_dict[alg],
                solver_factory=StochasticCluster.hill_climb_solver_factory)
            alg_dict['RandomHill_{0}'.format(alg)] = functools.partial(
                StochasticCluster.solve_with_initial_solver,
                init_solver=alg_dict[alg],
                solver_factory=StochasticCluster.random_hill_climb_solver_factory)

    @staticmethod
    def random_sparse_matrix(nodes: int) -> np.ndarray:
        graph = nx.gnp_random_graph(nodes, 0.3)
        for u, v in graph.edges():
            graph[u][v]['weight'] = np.random.randint(1, 50)
        return nx.to_numpy_array(graph)

    @staticmethod
    def scatter_plot(res_df, tests_num, timestamp):
        df = res_df.copy()
        df['color'] = df['algorithm'].astype("category")

        df.plot.scatter(x       = 'dataLoss_percentage',
                        y       = 'time_delta',
                        c       = 'color',
                        cmap    = "viridis",
                        s       = df['average_cluster_size']**2.5,
                        logy    = True,
                        alpha   = 0.5,
                        title   = "Clustering Comparison - Combined results\n(size = avg cluster size)",
                        grid    = True,
                        figsize = (18, 10))

        plt.savefig(os.path.join('output', f'scatter_full_{tests_num}_tests_{timestamp}.pdf'))
        plt.show(block=False)

    @staticmethod
    def box_plot(res_df, tests_num, timestamp):
        cols = ['algorithm', 'dataLoss_percentage', 'dataIn_percentage', 'time_delta', 'average_cluster_size']
        df = res_df[cols].copy()
        df['algorithm'] = df['algorithm'].astype("category")

        fig, ax_new = plt.subplots(2, 2, figsize=(18, 10), sharey='none')
        df.boxplot(by="algorithm", ax=ax_new, layout=(2, 2), grid=False, showmeans=True)
        fig.suptitle(f'Clustering Comparison - Combined results\n{tests_num} tests')
        plt.savefig(os.path.join('output', f'boxplot_combined_{tests_num}_tests_{timestamp}.pdf'))
        plt.show(block=False)

    @staticmethod
    def save_results_df(res_df, base_name):
        res_df.to_csv(base_name + '.csv')
        grouped_res_df = res_df.drop(columns=['test_num',
                                              'missions_num',
                                              'cluster_size',
                                              'agents_num']).groupby('algorithm')
        grouped_res_df.describe().to_csv(base_name + '_describe.csv')

    @staticmethod
    def filer_algorithms_to_compare(algo_filter_list, alg_dict):
        for key in algo_filter_list:
            if key in alg_dict:
                del alg_dict[key]


if __name__ == '__main__':

    test_mode = True  # else (False) - visualize mode

    # __ALL_ALGORITHMS = ['Greedy',
    #                     'GreedyLoop',
    #                     'Random',
    #                     'Blossom',
    #                     'Annealing_Greedy',
    #                     'Hill_Greedy',
    #                     'RandomHill_Greedy']

    if test_mode:
        multi_test_mode = True  # else (False) - single test mode

        if not multi_test_mode:
            # Run single config test
            ClusteringTest.test_demonstrate_testing_method(agents           = 36,
                                                           missions         = 1,
                                                           cluster_size     = 8,
                                                           tests            = 5,
                                                           algo_filter_list = ['GreedyLoop', 'Hill_Greedy'],
                                                           matrices_ids     = None)

        # Run multi-configs test
        else:
            ClusteringTest.multiple_parameters_testing_method(agents_num_list   = [16, 24],
                                                              cluster_size_list = [7],
                                                              missions          = 1,
                                                              num_tests         = 5,
                                                              algo_filter_list  = ['Hill_Greedy'])  # ['GreedyLoop', 'Hill_Greedy', 'Random'])

    else:
        # Visualize results
        results_csv_name = 'results_10_tests_2021_10_14-01:14:59_PM.csv'
        results_df = pd.read_csv(results_csv_name).drop(columns=['Unnamed: 0'])
        matrices_dir = 'matrices' + results_csv_name.split('results')[1].split('.csv')[0]
        crv = ClusteringResultViewer(results_df=results_df, matrices_dir=matrices_dir)

        # # Visualize result by index in csv
        # crv.show_single_result_by_index(100)

        while True:
            # Visualize random result by conditions
            wait_for = input()
            crv.show_random_single_result_by(algorithms               = ['Blossom'],    # see __ALL_ALGORITHMS
                                             agents_num               = [36, 48, 96],   # [24, 36, 48]
                                             cluster_size             = [7, 8, 15, 16],     # [4, 8, 16]
                                             max_data_loss_percentage = 1.0,            # [0 1]
                                             min_data_loss_percentage = 0.0,            # [0 1]
                                             max_time_delta_seconds   = 30.0,            # [0 1]
                                             min_time_delta_seconds   = 0.0)            # [0 1]

    print("Done")
