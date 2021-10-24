import unittest

import numpy as np

from ClusteringAlgs.LocalSearch import LocalSearchCluster
from RegretEstimators import DataLossEstimator


class test_LocalSearch(unittest.TestCase):

    def test_attempt_clustering_improvement_by_cluster_unification_clusters_cant_be_unified(self):
        matrix = np.zeros((4, 4))
        cluster_size = 3

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_unification = \
            LocalSearchCluster._attempt_clustering_improvement_by_cluster_unification(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertFalse(can_clustering_be_improved_by_unification)

    def test_attempt_clustering_improvement_by_cluster_unification_no_gain_by_unification(self):
        matrix = np.zeros((4, 4))
        cluster_size = 4

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_unification = \
            LocalSearchCluster._attempt_clustering_improvement_by_cluster_unification(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertFalse(can_clustering_be_improved_by_unification)

    def test_attempt_clustering_improvement_by_cluster_unification_unification_pays_off(self):
        matrix = np.ones((4, 4))
        cluster_size = 4

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_unification = \
            LocalSearchCluster._attempt_clustering_improvement_by_cluster_unification(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertTrue(can_clustering_be_improved_by_unification)

    def test_attempt_clustering_improvement_by_cluster_unification_cant_unify_the_same_cluster(self):
        matrix = np.ones((4, 4))
        cluster_size = 4

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_unification = \
            LocalSearchCluster._attempt_clustering_improvement_by_cluster_unification(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=1)

        self.assertFalse(can_clustering_be_improved_by_unification)

    def test_attempt_moving_agents_between_clusters_both_clusters_are_of_maximum_size(self):
        matrix = np.ones((4, 4))
        cluster_size = 2

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_moving_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertFalse(can_clustering_be_improved_by_moving_an_agent)

    def test_attempt_moving_agents_between_clusters_no_gain_by_agent_moving(self):
        matrix = np.zeros((4, 4))
        cluster_size = 4

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_moving_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertFalse(can_clustering_be_improved_by_moving_an_agent)

    def test_attempt_moving_agents_between_clusters_no_gain_by_agent_moving_2(self):
        matrix = np.array([[100.0, 100.0, 3.0, 3.0],
                           [100.0, 100.0, 3.0, 3.0],
                           [3.0, 3.0, 100.0, 100.0],
                           [3.0, 3.0, 100.0, 100.0]])
        cluster_size = 4

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_moving_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertFalse(can_clustering_be_improved_by_moving_an_agent)

    def test_attempt_moving_agents_between_clusters_cant_move_within_same_cluster(self):
        matrix = np.ones((4, 4))
        cluster_size = 4

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_moving_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=1)

        self.assertFalse(can_clustering_be_improved_by_moving_an_agent)

    def test_attempt_moving_agents_between_clusters_agents_are_moved_if_cluster_is_improved(self):
        matrix = np.ones((4, 4))
        cluster_size = 3

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_moving_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertTrue(can_clustering_be_improved_by_moving_an_agent)

    def test_attempt_switching_agents_between_clusters_agents_not_moved_in_the_same_cluster(self):
        matrix = np.array([[100.0, 100.0, 3.0, 3.0],
                           [100.0, 100.0, 3.0, 3.0],
                           [3.0, 3.0, 100.0, 100.0],
                           [3.0, 3.0, 100.0, 100.0]])
        cluster_size = 2

        agent_to_cluster_map = {0: 0, 1: 1, 2: 0, 3: 1}

        clustering = [{0, 2}, {1, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_switching_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertFalse(can_clustering_be_improved_by_moving_an_agent)

    def test_attempt_switching_agents_between_clusters_agents_not_moved_if_the_data_loss_is_not_improved(self):
        matrix = np.ones((4, 4))
        cluster_size = 2

        agent_to_cluster_map = {0: 0, 1: 0, 2: 1, 3: 1}

        clustering = [{0, 1}, {2, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_switching_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=0,
                second_agent_index=2)

        self.assertFalse(can_clustering_be_improved_by_moving_an_agent)

    def test_attempt_switching_agents_between_clusters_agents_are_switched_if_data_loss_is_improved(self):
        matrix = np.array([[100.0, 100.0, 3.0, 3.0],
                           [100.0, 100.0, 3.0, 3.0],
                           [3.0, 3.0, 100.0, 100.0],
                           [3.0, 3.0, 100.0, 100.0]])
        cluster_size = 2

        agent_to_cluster_map = {0: 0, 1: 1, 2: 0, 3: 1}

        clustering = [{0, 2}, {1, 3}]

        previous_data_loss = DataLossEstimator.calculate_data_loss(matrix=matrix, clusters=clustering)

        _, _, _, can_clustering_be_improved_by_moving_an_agent = \
            LocalSearchCluster._attempt_switching_agents_between_clusters(
                matrix=matrix,
                clustering=clustering,
                agent_to_cluster_map=agent_to_cluster_map,
                cluster_size=cluster_size,
                previous_data_loss_score=previous_data_loss,
                first_agent_index=1,
                second_agent_index=2)

        self.assertTrue(can_clustering_be_improved_by_moving_an_agent)

    def test_create_agent_to_clustering_map_clustering_does_not_contain_agent(self):
        clusters = [{1, 2, 3}, {4, 5}, {6}, {7, 8, 9}]  # agent 0 is missing

        with self.assertRaises(RuntimeError):
            LocalSearchCluster._create_agent_to_clustering_map(agents_num=10, initial_clustering=clusters)

    def test_create_agent_to_cluster_map_cluster_map_and_clustering_match(self):
        clusters = [{0}, {1, 2, 3}, {4, 5}, {6}, {7, 8, 9}]

        agent_to_cluster_map = LocalSearchCluster._create_agent_to_clustering_map(agents_num=10,
                                                                                  initial_clustering=clusters)

        # Check that clusters are contained in agent_to_cluster_map
        for cluster_index, cluster in enumerate(clusters):
            for agent in cluster:
                if agent_to_cluster_map[agent] != cluster_index:
                    self.fail(f'Agent {agent}\'s cluster by the cluster map does not match the actual cluster index.\n'
                              f'agent_to_cluster_map = {agent_to_cluster_map}\n'
                              f'agent_to_cluster_map[{agent}] = {agent_to_cluster_map[agent]}\n'
                              f'cluster_index = {cluster_index}\n'
                              f'clusters = {clusters}\n')

        for agent, cluster_index in agent_to_cluster_map.items():
            if agent not in clusters[cluster_index]:
                self.fail(f'Agent {agent}\'s cluster by the cluster map does not match the actual cluster index.\n'
                          f'agent_to_cluster_map = {agent_to_cluster_map}\n'
                          f'agent_to_cluster_map[{agent}] = {agent_to_cluster_map[agent]}\n'
                          f'cluster_index = {cluster_index}\n'
                          f'clusters = {clusters}\n')

    def test_create_agent_to_cluster_map_cluster_map_and_clustering_match_2(self):
        clusters = [{0, 4, 5}, {1, 2, 8}, {}, {6, 3}, {7, 9}]

        agent_to_cluster_map = LocalSearchCluster._create_agent_to_clustering_map(agents_num=10,
                                                                                  initial_clustering=clusters)

        # Check that clusters are contained in agent_to_cluster_map
        for cluster_index, cluster in enumerate(clusters):
            for agent in cluster:
                if agent_to_cluster_map[agent] != cluster_index:
                    self.fail(f'Agent {agent}\'s cluster by the cluster map does not match the actual cluster index.\n'
                              f'agent_to_cluster_map = {agent_to_cluster_map}\n'
                              f'agent_to_cluster_map[{agent}] = {agent_to_cluster_map[agent]}\n'
                              f'cluster_index = {cluster_index}\n'
                              f'clusters = {clusters}\n')

        for agent, cluster_index in agent_to_cluster_map.items():
            if agent not in clusters[cluster_index]:
                self.fail(f'Agent {agent}\'s cluster by the cluster map does not match the actual cluster index.\n'
                          f'agent_to_cluster_map = {agent_to_cluster_map}\n'
                          f'agent_to_cluster_map[{agent}] = {agent_to_cluster_map[agent]}\n'
                          f'cluster_index = {cluster_index}\n'
                          f'clusters = {clusters}\n')
