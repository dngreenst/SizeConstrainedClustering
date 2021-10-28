from typing import List, Set, Tuple
import numpy as np
from anytree import Node, search
from MatrixGenerators import  ReducedMatrix
from ClusteringAlgs.HierarchyCluster import MaxWeightCluster
from RegretEstimators import DataLossEstimator
import copy


def find_options(leaf: Node, cluster_size: int) -> List[Node]:
    leaf_size = len(leaf.cluster)
    curr: Node = leaf
    options = []
    while not curr.is_root and not options:
        for node in curr.siblings:
            options += [n for n in node.leaves if len(n.cluster) + leaf_size <= cluster_size]
        curr = curr.parent
    return options


def options_value(matrix: np.ndarray, node: Node, options: List[Node]) -> List[int]:
    return [ReducedMatrix.coarse_element(matrix, list(node.cluster), list(p.cluster))/(len(node.cluster) + len(p.cluster))
            for p in options]


def value_join_tree(matrix: np.ndarray, root: Node, cluster_size: int) -> List[Set[int]]:
    leaves = set(root.leaves)
    while leaves:
        closed_leaves = set()
        for leaf in sorted(list(leaves), key=lambda x: len(x.cluster), reverse=True): #type: Node
            options = find_options(leaf, cluster_size)
            if options:
                chosen: Node = options[np.argmax(options_value(matrix, leaf, options))]
                # add cluster from node to the leaf and remove node from tree
                chosen.parent = None
                leaf.cluster = leaf.cluster.union(chosen.cluster)
                closed_leaves.add(chosen)
            else:
                closed_leaves.add(leaf)
        leaves = leaves.difference(closed_leaves)
    return [node.cluster for node in root.leaves if node.cluster]


def join_tree(matrix: np.ndarray, root: Node, cluster_size: int) -> List[Set[int]]:
    leaves = set(root.leaves)
    while leaves:
        closed_leaves = set()
        for leaf in sorted(list(leaves), key=lambda x: len(x.cluster), reverse=True): #type: Node
            options = find_options(leaf, cluster_size)
            if options:
                chosen: Node = max(options, key=lambda x: len(x.cluster))
                # add cluster from node to the leaf and remove node from tree
                chosen.parent = None
                leaf.cluster = leaf.cluster.union(chosen.cluster)
                closed_leaves.add(chosen)
            else:
                closed_leaves.add(leaf)
        leaves = leaves.difference(closed_leaves)
    return [node.cluster for node in root.leaves if node.cluster]


def random_join_tree(root: Node, cluster_size: int) -> List[Set[int]]:
    leaves = set(root.leaves)
    while leaves:
        closed_leaves = set()
        for leaf in np.random.permutation(list(leaves)): #type: Node
            options = find_options(leaf, cluster_size)
            if options:
                chosen: Node = np.random.choice(options)
                # add cluster from node to the leaf and remove node from tree
                chosen.parent = None
                leaf.cluster = leaf.cluster.union(chosen.cluster)
                closed_leaves.add(chosen)
            else:
                closed_leaves.add(leaf)
        leaves = leaves.difference(closed_leaves)
    return [node.cluster for node in root.leaves if node.cluster]


def max_random_join(matrix: np.ndarray, root: Node, cluster_size: int) -> List[Set[int]]:
    min_regret = np.inf
    best_clusters = []
    for i in range(100):
        clusters = random_join_tree(copy.deepcopy(root), cluster_size)
        regret = DataLossEstimator.calculate_data_loss(matrix, clusters)
        if regret < min_regret:
            min_regret = regret
            best_clusters = clusters
    return best_clusters




