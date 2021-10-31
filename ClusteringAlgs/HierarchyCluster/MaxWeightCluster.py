from typing import List, Set, Tuple
import math
import numpy as np
import networkx as nx
from ClusteringAlgs.LocalSearch import LocalSearchCluster
from MatrixGenerators import ReducedMatrix


class MaxWeightCluster:
    @staticmethod
    def __maxweight_match(matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Mapping vertex (indexes in matrix) to matches"""
        graph: nx.Graph = nx.from_numpy_matrix(matrix)
        matches = list(nx.max_weight_matching(graph))
        return matches

    @staticmethod
    def __choose_matches(matrix: np.ndarray, matches: List[Tuple[int, int]], matches_percent: float = 0.50) -> List[Tuple[int, int]]:
        matches_weights = [(matches[i], matrix[(matches[i])[0]][(matches[i])[1]]) for i in range(len(matches))]
        matches_weights.sort(key=lambda tup: tup[1], reverse=True)
        number_of_matches = len(matches_weights)
        if number_of_matches > 0:
            mean = sum([matches_weights[i][1] for i in range(number_of_matches)]) / number_of_matches
            percentile = matches_weights[-math.floor((1 - matches_percent) * number_of_matches)][1]
            if mean > percentile:
                matches_weights = list(filter(lambda i: i[1] >= mean, matches_weights))
            else:
                matches_weights = matches_weights[:-math.floor((1 - matches_percent) * number_of_matches) or None]

        matches = [match[0] for match in matches_weights]
        return matches

    @staticmethod
    def __create_unifications_list(matrix: np.ndarray, chosen_matches: List[Tuple[int, int]]):
        unifications_list = chosen_matches.copy()
        all_nodes_set = set(np.arange(matrix.shape[0]))
        matched_list = [set(m) for m in chosen_matches]
        matched_set = set.union(*matched_list) if len(matched_list) > 0 else set()
        for node in all_nodes_set.difference(matched_set):
            unifications_list.append((node, node))
        return unifications_list

    @staticmethod
    def group_clusters(clusters: List[Set[int]], groups_by_index: List) -> List[Set[int]]:
        new_clusters = []
        for group in groups_by_index:
            new_clusters.append(set.union(*[clusters[i] for i in group]))
        return new_clusters

    @staticmethod
    def cluster_once(clusters: List[Set[int]],
                     matrix: np.ndarray) -> Tuple[List[Set[int]], List[Tuple[int, int]], bool]:
        matches = MaxWeightCluster.__maxweight_match(matrix)
        chosen_matches = MaxWeightCluster.__choose_matches(matrix, matches, matches_percent=0.50)
        current_iter_unifications = MaxWeightCluster.__create_unifications_list(matrix, chosen_matches)
        clusters = MaxWeightCluster.group_clusters(clusters, current_iter_unifications)
        cluster_was_successful = len(chosen_matches) > 0
        return clusters, current_iter_unifications, cluster_was_successful

    @staticmethod
    def cluster(matrix: np.ndarray, cluster_size: int) -> List[Set[int]]:
        current_matrix = matrix.copy()
        clusters = [{i} for i in range(current_matrix.shape[0])]
        while True:
            clusters, matches, cluster_was_successful = MaxWeightCluster.cluster_once(clusters, current_matrix)
            if not cluster_was_successful:
                break
            current_matrix = ReducedMatrix.coarse_matrix(current_matrix, matches, 1.0)
            for i in range(current_matrix.shape[0]):
                for j in range(i):
                    if len(clusters[i]) + len(clusters[j]) > cluster_size:
                        current_matrix[i][j] = 0
                        current_matrix[j][i] = 0
        clusters = LocalSearchCluster.local_search_clustering(matrix=matrix,
                                                              cluster_size=cluster_size,
                                                              initial_clustering=clusters)
        return clusters


def maxweight_cluster_with_missions(block_matrix: np.ndarray, n: int, m: int, cluster_size: int) -> List[Set[int]]:
    matrix = ReducedMatrix.reduce_block_matrix(block_matrix, n, m, cluster_size)
    return MaxWeightCluster.cluster(matrix, cluster_size)
