from typing import List, Set, Tuple
import numpy as np
import networkx as nx

from MatrixGenerators import ReducedMatrix


class BlossomCluster:
    @staticmethod
    def __blossom_match(matrix: np.ndarray) -> List[Tuple[int, ...]]:
        """Mapping vertex (indexes in matrix) to matches"""
        graph: nx.Graph = nx.from_numpy_matrix(matrix)
        matches = list(nx.max_weight_matching(graph))
        if matrix.shape[0] % 2 != 0:
            sum_matched_indexes = np.sum(np.asarray(matches).flatten())
            sum_all_indexes = np.sum(np.arange(matrix.shape[0]))
            missing_index = sum_all_indexes - sum_matched_indexes
            matches.append((missing_index, missing_index))
        return matches

    @staticmethod
    def group_clusters(clusters: List[Set[int]], groups_by_index: List) -> List[Set[int]]:
        new_clusters = []
        for group in groups_by_index:
            new_clusters.append(set.union(*[clusters[i] for i in group]))
        return new_clusters

    @staticmethod
    def cluster_once(clusters: List[Set[int]], matrix: np.ndarray) -> Tuple[List[Set[int]], List[Tuple[int, ...]]]:
        matches = BlossomCluster.__blossom_match(matrix)
        return BlossomCluster.group_clusters(clusters, matches), matches

    @staticmethod
    def cluster(matrix: np.array, cluster_size: int) -> List[Set[int]]:
        n = matrix.shape[0]
        k = cluster_size
        clusters = [{i} for i in range(n)]
        while True:
            clusters, matches = BlossomCluster.cluster_once(clusters, matrix)
            k /= 2
            if round(k) <= 1:
                break;
            matrix = ReducedMatrix.coarse_matrix(matrix, matches, 1.0)
        return clusters


def blossom_cluster_with_missions(block_matrix: np.array, n: int, m: int, cluster_size: int) -> List[Set[int]]:
    matrix = ReducedMatrix.reduce_block_matrix(block_matrix, n, m, cluster_size)
    return BlossomCluster.cluster(matrix, cluster_size)


