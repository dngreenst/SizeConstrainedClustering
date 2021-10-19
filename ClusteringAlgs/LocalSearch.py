import copy
from typing import Optional, List, Set, Tuple, Dict

import numpy as np

from RegretEstimators import DataLossEstimator


class LocalSearchCluster:

    @staticmethod
    def local_search_clustering(matrix: np.ndarray,
                                cluster_size: int,
                                initial_clustering: Optional[List[Set[int]]]) -> List[Set[int]]:
        agents_num = matrix.shape[0]

        clustering_might_be_improvable = True

        if initial_clustering is None:
            # If no clustering was provided, create a clustering of singletons.
            initial_clustering = []
            for agent_index in range(agents_num):
                initial_clustering.append({agent_index})

        current_agent_to_cluster_map = LocalSearchCluster._create_agent_to_clustering_map(
            agents_num=agents_num,
            initial_clustering=initial_clustering)

        current_clustering = initial_clustering

        while clustering_might_be_improvable:
            clustering_might_be_improvable = False

            for first_agent_index in range(agents_num):
                for second_agent_index in range(first_agent_index):
                    new_clustering, new_agent_to_cluster_map, was_unification_successful = \
                        LocalSearchCluster._attempt_clustering_improvement_by_cluster_unification(
                            matrix=matrix,
                            clustering=current_clustering,
                            agent_to_cluster_map=current_agent_to_cluster_map,
                            cluster_size=cluster_size,
                            first_agent_index=first_agent_index,
                            second_agent_index=second_agent_index)

                    if was_unification_successful:
                        current_clustering = new_clustering
                        current_agent_to_cluster_map = new_agent_to_cluster_map
                        clustering_might_be_improvable = True
                        break

                    new_clustering, new_agent_to_cluster_map, was_moving_agents_between_clusters_successful = \
                        LocalSearchCluster._attempt_moving_agents_between_clusters(
                            matrix=matrix,
                            clustering=current_clustering,
                            agent_to_cluster_map=current_agent_to_cluster_map,
                            cluster_size=cluster_size,
                            first_agent_index=first_agent_index,
                            second_agent_index=second_agent_index)

                    if was_moving_agents_between_clusters_successful:
                        current_clustering = new_clustering
                        current_agent_to_cluster_map = new_agent_to_cluster_map
                        clustering_might_be_improvable = True
                        break

                    new_clustering, new_agent_to_cluster_map, was_switching_agents_between_clusters_successful = \
                        LocalSearchCluster._attempt_switching_agents_between_clusters(
                            matrix=matrix,
                            clustering=current_clustering,
                            agent_to_cluster_map=current_agent_to_cluster_map,
                            cluster_size=cluster_size,
                            first_agent_index=first_agent_index,
                            second_agent_index=second_agent_index)

                    if was_switching_agents_between_clusters_successful:
                        current_clustering = new_clustering
                        current_agent_to_cluster_map = new_agent_to_cluster_map
                        clustering_might_be_improvable = True
                        break

        return current_clustering

    @staticmethod
    def _create_agent_to_clustering_map(agents_num, initial_clustering) -> Dict[int, int]:
        agent_to_cluster_map: Dict[int, int] = {}
        for agent_index in range(agents_num):
            for cluster_index, cluster in enumerate(initial_clustering):
                if agent_index in cluster:
                    agent_to_cluster_map[agent_index] = cluster_index
                    break
            if agent_index not in agent_to_cluster_map.keys():
                raise RuntimeError(f'Unexpectedly, there is no cluster for agent {agent_index} in {initial_clustering}')

        return agent_to_cluster_map

    @staticmethod
    def _attempt_clustering_improvement_by_cluster_unification(matrix: np.array,
                                                               clustering: List[Set[int]],
                                                               agent_to_cluster_map: Dict[int, int],
                                                               cluster_size: int,
                                                               first_agent_index: int,
                                                               second_agent_index: int) -> \
            Tuple[List[Set[int]], Dict[int, int], bool]:

        # Extract current agent clusters
        first_agent_cluster = clustering[agent_to_cluster_map[first_agent_index]]
        second_agent_cluster = clustering[agent_to_cluster_map[second_agent_index]]

        # Check whether cluster can be unified at all
        if len(first_agent_cluster) + len(second_agent_cluster) <= cluster_size:

            # If clusters can be unified, check whether this improves the utility.
            score_before_unification = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)
            clustering_after_unification = copy.deepcopy(clustering)
            clustering_after_unification[agent_to_cluster_map[first_agent_index]] = \
                first_agent_cluster.union(second_agent_cluster)
            clustering_after_unification.pop(agent_to_cluster_map[second_agent_index])
            score_after_unification = DataLossEstimator.calculate_data_loss(matrix=matrix,
                                                                            clusters=clustering_after_unification)

            if score_after_unification < score_before_unification:
                # If the unification improved the utility, we need to generate a new agent_to_cluster_map and return
                # the new clustering.
                new_agent_to_cluster_map = \
                    LocalSearchCluster._create_agent_to_clustering_map(agents_num=matrix.shape[0],
                                                                       initial_clustering=clustering_after_unification)
                return clustering_after_unification, new_agent_to_cluster_map, True

        # If the unification is not possible, or doesn't improve the clustering, return the previous clustering.
        return clustering, agent_to_cluster_map, False

    @staticmethod
    def _attempt_moving_agents_between_clusters(matrix: np.array,
                                                clustering: List[Set[int]],
                                                agent_to_cluster_map: Dict[int, int],
                                                cluster_size: int,
                                                first_agent_index: int,
                                                second_agent_index: int) -> Tuple[List[Set[int]], Dict[int, int], bool]:
        pass

    @staticmethod
    def _attempt_switching_agents_between_clusters(matrix: np.array,
                                                   clustering: List[Set[int]],
                                                   agent_to_cluster_map: Dict[int, int],
                                                   cluster_size: int,
                                                   first_agent_index: int,
                                                   second_agent_index: int) -> \
            Tuple[List[Set[int]], Dict[int, int], bool]:
        pass
