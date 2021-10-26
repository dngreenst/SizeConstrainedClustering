from typing import List, Set, Tuple
import math
import numpy as np
import networkx as nx

from MatrixGenerators import ReducedMatrix


def is_equal(l1, l2):
    l1.sort(key=len)
    l2.sort(key=len)
    if l1 == l2:
        return True
    return False


class BlossomCluster:
    @staticmethod
    def __blossom_match(matrix: np.ndarray, matches_percent: float = 0.50) -> List[Tuple[int, int]]:
        """Mapping vertex (indexes in matrix) to matches"""
        # init
        graph: nx.Graph = nx.from_numpy_matrix(matrix)
        all_nodes = set(np.arange(matrix.shape[0]))

        # get matches and weights
        matches = list(nx.max_weight_matching(graph))
        matches_weights = [(matches[i], matrix[(matches[i])[0]][(matches[i])[1]]) for i in range(len(matches))]
        matches_weights.sort(key=lambda tup: tup[1], reverse=True)

        # get only matches that has more value then max(
        number_of_matches = len(matches_weights)
        if number_of_matches > 0:
            mean = sum([matches_weights[i][1] for i in range(number_of_matches)]) / number_of_matches
            percentile = matches_weights[-math.floor((1 - matches_percent) * number_of_matches)][1]
            if mean > percentile:
                matches_weights = list(filter(lambda i: i[1] >= mean, matches_weights))
            else:
                matches_weights = matches_weights[:-math.floor((1 - matches_percent) * number_of_matches) or None]

        matches = [match[0] for match in matches_weights]

        # if no matches after filter, return non-matched
        matched_list = [set(m[0]) for m in matches_weights]
        if not matched_list:
            for node in all_nodes:
                matches.append((node, node))
            return matches

        # else - return matched
        matched = set.union(*matched_list)
        if len(matched) != len(all_nodes):
            for node in all_nodes.difference(matched):
                matches.append((node, node))

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
    def cluster(matrix: np.ndarray, cluster_size: int) -> List[Set[int]]:
        clusters = [{i} for i in range(matrix.shape[0])]
        while True:
            matches_init = [(i, i) for i in range(matrix.shape[0])]
            clusters, matches = BlossomCluster.cluster_once(clusters, matrix)
            if is_equal(matches, matches_init):
                break
            matrix = ReducedMatrix.coarse_matrix(matrix, matches, 1.0)
            for i in range(matrix.shape[0]):
                for j in range(i):
                    if len(clusters[i]) + len(clusters[j]) > cluster_size:
                        matrix[i][j] = 0
                        matrix[j][i] = 0
        return clusters


def blossom_cluster_with_missions(block_matrix: np.ndarray, n: int, m: int, cluster_size: int) -> List[Set[int]]:
    matrix = ReducedMatrix.reduce_block_matrix(block_matrix, n, m, cluster_size)
    return BlossomCluster.cluster(matrix, cluster_size)
