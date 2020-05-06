import unittest
from ClusteringAlgsComparisons.ClusteringComparison import ClusteringComparator
from ClusteringAlgs import RandomAssignment
from ClusteringAlgs.HierarchyCluster import BlossomCluster
from ClusteringAlgs.HierarchyCluster import FidlerVecCluster



class ClusteringTest(unittest.TestCase):
    @staticmethod
    def test_demonstrate_testing_method():
        comparator = ClusteringComparator(agents_num=100, missions_num=5, cluster_size=13, tests_num=5)
        alg_dict = dict()
        alg_dict["Random"] = RandomAssignment.random_permutation_clustering_from_matrix
        alg_dict["Blossom"] = BlossomCluster.BlossomCluster.cluster
        alg_dict["Fidler"] = FidlerVecCluster.fidler_cluster

        comparator.compare(alg_dict)
        comparator.show_data()
