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

        self.assertTrue(can_clustering_be_improved_by_moving_an_agent)
