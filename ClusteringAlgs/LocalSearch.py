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

        curr_data_loss_score = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=current_clustering)

        while clustering_might_be_improvable:
            clustering_might_be_improvable = False

            for first_agent_index in range(agents_num):
                for second_agent_index in range(first_agent_index):
                    new_clustering, new_agent_to_cluster_map, data_loss_after_unification, was_unification_successful = \
                        LocalSearchCluster._attempt_clustering_improvement_by_cluster_unification(
                            matrix=matrix,
                            clustering=current_clustering,
                            agent_to_cluster_map=current_agent_to_cluster_map,
                            cluster_size=cluster_size,
                            previous_data_loss_score=curr_data_loss_score,
                            first_agent_index=first_agent_index,
                            second_agent_index=second_agent_index)

                    if was_unification_successful:
                        current_clustering = new_clustering
                        current_agent_to_cluster_map = new_agent_to_cluster_map
                        curr_data_loss_score = data_loss_after_unification
                        clustering_might_be_improvable = True
                        break

                    new_clustering, new_agent_to_cluster_map, data_loss_after_moving_an_agent, was_moving_agents_between_clusters_successful = \
                        LocalSearchCluster._attempt_moving_agents_between_clusters(
                            matrix=matrix,
                            clustering=current_clustering,
                            agent_to_cluster_map=current_agent_to_cluster_map,
                            cluster_size=cluster_size,
                            previous_data_loss_score=curr_data_loss_score,
                            first_agent_index=first_agent_index,
                            second_agent_index=second_agent_index)

                    if was_moving_agents_between_clusters_successful:
                        current_clustering = new_clustering
                        current_agent_to_cluster_map = new_agent_to_cluster_map
                        curr_data_loss_score = data_loss_after_moving_an_agent
                        clustering_might_be_improvable = True
                        break

                    new_clustering, new_agent_to_cluster_map, data_loss_after_switching_between_agents, was_switching_agents_between_clusters_successful = \
                        LocalSearchCluster._attempt_switching_agents_between_clusters(
                            matrix=matrix,
                            clustering=current_clustering,
                            agent_to_cluster_map=current_agent_to_cluster_map,
                            cluster_size=cluster_size,
                            previous_data_loss_score=curr_data_loss_score,
                            first_agent_index=first_agent_index,
                            second_agent_index=second_agent_index)

                    if was_switching_agents_between_clusters_successful:
                        current_clustering = new_clustering
                        current_agent_to_cluster_map = new_agent_to_cluster_map
                        curr_data_loss_score = data_loss_after_switching_between_agents
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
                                                               previous_data_loss_score: float,
                                                               first_agent_index: int,
                                                               second_agent_index: int) -> \
            Tuple[List[Set[int]], Dict[int, int], float, bool]:

        # Extract current agent clusters
        first_agent_cluster = clustering[agent_to_cluster_map[first_agent_index]]
        second_agent_cluster = clustering[agent_to_cluster_map[second_agent_index]]

        # Check whether cluster can be unified at all
        if len(first_agent_cluster) + len(second_agent_cluster) <= cluster_size:

            # If clusters can be unified, check whether this improves the utility.
            clustering_after_unification = copy.deepcopy(clustering)
            clustering_after_unification[agent_to_cluster_map[first_agent_index]] = \
                first_agent_cluster.union(second_agent_cluster)
            clustering_after_unification.pop(agent_to_cluster_map[second_agent_index])
            score_after_unification = DataLossEstimator.calculate_data_loss(matrix=matrix,
                                                                            clusters=clustering_after_unification)

            if score_after_unification < previous_data_loss_score:
                # If the unification improved the utility, we need to generate a new agent_to_cluster_map and return
                # the new clustering.
                new_agent_to_cluster_map = \
                    LocalSearchCluster._create_agent_to_clustering_map(agents_num=matrix.shape[0],
                                                                       initial_clustering=clustering_after_unification)
                return clustering_after_unification, new_agent_to_cluster_map, score_after_unification, True

        # If the unification is not possible, or doesn't improve the clustering, return the previous clustering.
        return clustering, agent_to_cluster_map, previous_data_loss_score, False

    @staticmethod
    def _attempt_moving_agents_between_clusters(matrix: np.array,
                                                clustering: List[Set[int]],
                                                agent_to_cluster_map: Dict[int, int],
                                                cluster_size: int,
                                                previous_data_loss_score: float,
                                                first_agent_index: int,
                                                second_agent_index: int) -> Tuple[
        List[Set[int]], Dict[int, int], float, bool]:
        # Extract current agent clusters
        first_agent_cluster = clustering[agent_to_cluster_map[first_agent_index]]
        second_agent_cluster = clustering[agent_to_cluster_map[second_agent_index]]

        score_after_moving_first_agent_to_seconds_cluster = -np.inf
        score_after_moving_second_agent_to_firsts_cluster = -np.inf

        clustering_after_moving_second_agent_to_firsts_cluster = []
        clustering_after_moving_first_agent_to_seconds_cluster = []

        # Check whether second agent can be moved into first agent's cluster
        if len(first_agent_cluster) < cluster_size:
            # If the second agent can be moved, generate the new clustering and calculate the new data loss score
            clustering_after_moving_second_agent_to_firsts_cluster = copy.deepcopy(clustering)
            clustering_after_moving_second_agent_to_firsts_cluster[agent_to_cluster_map[first_agent_index]].add(
                second_agent_index)
            clustering_after_moving_second_agent_to_firsts_cluster[agent_to_cluster_map[second_agent_index]].remove(
                second_agent_index)
            score_after_moving_second_agent_to_firsts_cluster = \
                DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        # Check whether the first agent can be moved into the second agent's cluster
        if len(second_agent_cluster) < cluster_size:
            # If the first agent can be moved, generate the new clustering and calculate the new data loss score
            clustering_after_moving_first_agent_to_seconds_cluster = copy.deepcopy(clustering)
            clustering_after_moving_first_agent_to_seconds_cluster[agent_to_cluster_map[second_agent_index]].add(
                first_agent_index)
            clustering_after_moving_first_agent_to_seconds_cluster[agent_to_cluster_map[first_agent_index]].remove(
                first_agent_index)
            score_after_moving_first_agent_to_seconds_cluster = \
                DataLossEstimator.calculate_data_loss(matrix=matrix,
                                                      clusters=clustering_after_moving_first_agent_to_seconds_cluster)

        # Check whether either move improved the clustering
        if previous_data_loss_score < min(score_after_moving_first_agent_to_seconds_cluster,
                                          score_after_moving_second_agent_to_firsts_cluster):
            # If no move was possible, or if neither improved the clustering, return the original clustering.
            return clustering, agent_to_cluster_map, previous_data_loss_score, False

        # One of the new clusterings improved the data loss - return the best one.
        if score_after_moving_second_agent_to_firsts_cluster < score_after_moving_first_agent_to_seconds_cluster:
            agent_to_cluster_map[second_agent_index] = agent_to_cluster_map[first_agent_index]
            return clustering_after_moving_second_agent_to_firsts_cluster, agent_to_cluster_map, score_after_moving_second_agent_to_firsts_cluster, True

        agent_to_cluster_map[first_agent_index] = agent_to_cluster_map[second_agent_index]
        return clustering_after_moving_first_agent_to_seconds_cluster, agent_to_cluster_map, score_after_moving_first_agent_to_seconds_cluster, True

    @staticmethod
    def _attempt_switching_agents_between_clusters(matrix: np.array,
                                                   clustering: List[Set[int]],
                                                   agent_to_cluster_map: Dict[int, int],
                                                   cluster_size: int,
                                                   previous_data_loss_score: float,
                                                   first_agent_index: int,
                                                   second_agent_index: int) -> \
            Tuple[List[Set[int]], Dict[int, int], float, bool]:
        original_first_agent_cluster_index = agent_to_cluster_map[first_agent_index]
        original_second_agent_cluster_index = agent_to_cluster_map[second_agent_index]

        clustering_after_agent_exchange = copy.deepcopy(clustering)

        # Remove the first agent from its original cluster, and add the second to it.
        clustering_after_agent_exchange[original_first_agent_cluster_index].remove(first_agent_index)
        clustering_after_agent_exchange[original_first_agent_cluster_index].add(second_agent_index)

        # Remove the second agent from its original cluster, and add the first to it.
        clustering_after_agent_exchange[original_second_agent_cluster_index].remove(second_agent_index)
        clustering_after_agent_exchange[original_second_agent_cluster_index].add(first_agent_index)

        data_loss_after_exchange = DataLossEstimator.calculate_data_loss(matrix=matrix,
                                                                         clusters=clustering_after_agent_exchange)

        if data_loss_after_exchange < previous_data_loss_score:
            agent_to_cluster_map[first_agent_index] = original_second_agent_cluster_index
            agent_to_cluster_map[second_agent_index] = original_first_agent_cluster_index

            return clustering_after_agent_exchange, agent_to_cluster_map, data_loss_after_exchange, True

        return clustering, agent_to_cluster_map, previous_data_loss_score, False
