import unittest
from ClusteringAlgsComparisons.ClusteringComparison import ClusteringComparator
from ClusteringAlgs import RandomAssignment
from ClusteringAlgs.HierarchyCluster import BlossomCluster
from ClusteringAlgs.HierarchyCluster import FidlerVecCluster
from ClusteringAlgs.HierarchyCluster import SpectralClustering
import numpy as np
import functools
import math





class ClusteringTest:
    @staticmethod
    def test_demonstrate_testing_method():
        comparator = ClusteringComparator(agents_num=100, missions_num=5, cluster_size=13, tests_num=10)
        alg_dict = dict()
        #alg_dict["Random"] = RandomAssignment.random_permutation_clustering_from_matrix
        alg_dict["Blossom"] = BlossomCluster.BlossomCluster.cluster
        n = 5
        fidler_method_dict = dict()
        fidler_method_dict["agglom"] = FidlerVecCluster.agglomerative_partition
        fidler_method_dict["kmeans"] = FidlerVecCluster.k_means_partition
        ClusteringTest.add_fidler_algs(alg_dict, fidler_method_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        #ClusteringTest.add_spectral_algs(alg_dict, np.arange(1, n + 1), np.arange(2, n + 1))
        comparator.compare(alg_dict)
        comparator.show_data()

    @staticmethod
    def add_fidler_algs(alg_dict, method_dict: dict, num_vectors: np.ndarray, num_clusters: np.ndarray):
        method_arr = list(method_dict.keys())
        fidler_options = np.array(np.meshgrid(num_vectors, num_clusters, method_arr)).T.reshape(-1, 3)
        for t in fidler_options:
            alg_dict["Fidler_{2}, vec: {0}, clust: {1}".format(t[0], t[1], t[2])] = \
                functools.partial(FidlerVecCluster.fidler_cluster, num_fidler_vectors=int(t[0]), num_clusters=int(t[1]),
                                  method_function=method_dict[t[2]])

    @staticmethod
    def add_spectral_algs(alg_dict: dict, num_vectors: np.ndarray, num_clusters: np.ndarray):
        spectral_options = np.array(np.meshgrid(num_vectors, num_clusters)).T.reshape(-1, 2)
        for t in spectral_options:
            alg_dict["Spectral, vec: {0}, clust: {1}".format(t[0], t[1])] = \
                functools.partial(SpectralClustering.cluster, num_fidler_vectors=int(t[0]), num_clusters=int(t[1]))


ClusteringTest.test_demonstrate_testing_method()

