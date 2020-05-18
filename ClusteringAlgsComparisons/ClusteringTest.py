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




class ClusteringTest:
    @staticmethod
    def test_demonstrate_testing_method():
        agents = 100
        missions = 1
        cluster_size = 8
        tests = 10

        join_method_dict = dict()
        join_method_dict["random"] = TreeJoin.max_random_join
        join_method_dict["size"] = TreeJoin.join_tree
        join_method_dict["value"] = TreeJoin.value_join_tree

        """"""
        comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
                                          tests_num=tests)
        alg_dict = dict()
        alg_dict["Random"] = RandomAssignment.random_permutation_clustering_from_matrix
        alg_dict["Blossom"] = BlossomCluster.BlossomCluster.cluster

        fidler_method_dict = dict()
        fidler_method_dict["agglom"] = FidlerVecCluster.agglomerative_partition
        fidler_method_dict["kmeans"] = FidlerVecCluster.k_means_partition
        ClusteringTest.add_fidler_algs(alg_dict, fidler_method_dict, join_method_dict, 2, 2)
        ClusteringTest.add_spectral_algs(alg_dict, join_method_dict, 2, 2)
        comparator.compare(alg_dict)
        comparator.show_data()
        """"""
        comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
                                          tests_num=tests, identifier=1)
        n = 3
        alg_dict = dict()
        fidler_method_dict = dict()
        fidler_method_dict["agglom"] = FidlerVecCluster.agglomerative_partition
        ClusteringTest.add_fidler_algs(alg_dict, fidler_method_dict, join_method_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        comparator.compare(alg_dict)
        comparator.show_data()
        """"""
        comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
                                          tests_num=tests, identifier=2)
        n = 3
        alg_dict = dict()
        fidler_method_dict = dict()
        fidler_method_dict["kmeans"] = FidlerVecCluster.k_means_partition
        ClusteringTest.add_fidler_algs(alg_dict, fidler_method_dict, join_method_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        comparator.compare(alg_dict)
        comparator.show_data()
        """"""
        comparator = ClusteringComparator(agents_num=agents, missions_num=missions, cluster_size=cluster_size,
                                          tests_num=tests, identifier=3)
        n = 3
        alg_dict = dict()
        ClusteringTest.add_spectral_algs(alg_dict, join_method_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        comparator.compare(alg_dict)
        comparator.show_data()
        """"""
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


ClusteringTest.test_demonstrate_testing_method()

