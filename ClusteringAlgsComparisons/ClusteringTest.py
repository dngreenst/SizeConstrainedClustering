import unittest
from ClusteringAlgsComparisons.ClusteringComparison import ClusteringComparator
from ClusteringAlgs import RandomAssignment
from ClusteringAlgs.HierarchyCluster import BlossomCluster
from ClusteringAlgs.HierarchyCluster import FidlerVecCluster
from ClusteringAlgs.HierarchyCluster import SpectralClustering
import numpy as np
import functools
import math
import matplotlib.pyplot as plt
from ClusteringAlgs.HierarchyCluster import TreeJoin
from ClusteringAlgs.StochasticCluster import StochasticCluster

import networkx as nx


class ClusteringTest:
    @staticmethod
    def test_demonstrate_testing_method():
        agents = 24
        missions = 1
        cluster_size = 8
        tests = 50

        join_method_dict = dict()
        join_method_dict["random"] = TreeJoin.max_random_join
        join_method_dict["size"] = TreeJoin.join_tree
        join_method_dict["value"] = TreeJoin.value_join_tree

        fidler_method_dict = dict()
        fidler_method_dict["agglom"] = FidlerVecCluster.agglomerative_partition
        fidler_method_dict["kmeans"] = FidlerVecCluster.k_means_partition
        """"""
        comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
                                          tests_num=tests)
        alg_dict = dict()
        alg_dict["Random"] = RandomAssignment.random_permutation_clustering_from_matrix
        alg_dict["Blossom"] = BlossomCluster.BlossomCluster.cluster
        ClusteringTest.add_stochastic_algs(alg_dict)
        # ClusteringTest.add_fidler_algs(alg_dict, fidler_method_dict, join_method_dict, 2, 2)
        # ClusteringTest.add_spectral_algs(alg_dict, join_method_dict, 2, 2)

        comparator.compare(alg_dict)
        comparator.show_data()
        """"""
        # comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
        #                                   tests_num=tests, identifier=1)
        # n = 3
        # alg_dict = dict()
        # fidler_method_dict = dict()
        # fidler_method_dict["agglom"] = FidlerVecCluster.agglomerative_partition
        # ClusteringTest.add_fidler_algs(alg_dict, fidler_method_dict, join_method_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        # comparator.compare(alg_dict)
        # comparator.show_data()
        # """"""
        # comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
        #                                   tests_num=tests, identifier=2)
        # n = 3
        # alg_dict = dict()
        # fidler_method_dict = dict()
        # fidler_method_dict["kmeans"] = FidlerVecCluster.k_means_partition
        # ClusteringTest.add_fidler_algs(alg_dict, fidler_method_dict, join_method_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        # comparator.compare(alg_dict)
        # comparator.show_data()
        # """"""
        # comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
        #                                   tests_num=tests, identifier=3)
        # n = 3
        # alg_dict = dict()
        # ClusteringTest.add_spectral_algs(alg_dict, join_method_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        # comparator.compare(alg_dict)
        # comparator.show_data()
        # """"""
        plt.show()

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
        # alg_dict['Genetic'] = functools.partial(StochasticCluster.solve,
        #                                        solver=StochasticCluster.genetic_solver_factory())
        #alg_dict['Annealing'] = functools.partial(StochasticCluster.solve,
        #                                          solver=StochasticCluster.annealing_solver_factory())
        #alg_dict['Hill'] = functools.partial(StochasticCluster.solve,
        #                                     solver=StochasticCluster.hill_climb_solver_factory())
        #alg_dict['RandomHill'] = functools.partial(StochasticCluster.solve,
        #                                           solver=StochasticCluster.random_hill_climb_solver_factory())
        # for alg in ['Greedy', 'GreedyLoop']: #, 'Blossom']:
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









ClusteringTest.test_demonstrate_testing_method()

