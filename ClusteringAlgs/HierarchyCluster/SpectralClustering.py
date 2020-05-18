import networkx as nx
from typing import List, Set, Tuple
import numpy as np
import sklearn.cluster as skc
from anytree import Node, search
from ClusteringAlgs.HierarchyCluster import TreeJoin


ZERO = 0.00001


def is_graph_connected(laplacian_eig_values: np.ndarray) -> bool:
    # graph is not connected if the second lowest eigenvalue of the laplacian matrix is 0
    zero_eig_value_qty = np.sum(np.abs(laplacian_eig_values) < ZERO)
    return True if zero_eig_value_qty < 2 else False


def partition(graph: nx.Graph, num_fidler_vectors: int = 1, num_clusters: int = 2) -> List[Set[int]]:
    laplacian = nx.laplacian_matrix(graph).toarray()
    eig_values, eig_vectors = np.linalg.eig(laplacian)
    if is_graph_connected(eig_values):
        matrix = nx.to_numpy_array(graph)
        assignment = skc.spectral_clustering(matrix, num_clusters, num_fidler_vectors)
        node_labels = np.asarray(graph.nodes)
        non_labeled_clusters = [np.flatnonzero(assignment == i) for i in np.unique(assignment)]
        clusters = [set(node_labels[cluster]) for cluster in non_labeled_clusters]
    else:
        clusters = [set(nx.subgraph(graph, g).nodes) for g in nx.connected_components(graph)]
    return clusters


def cluster(matrix: np.ndarray, cluster_size: int, join_function=TreeJoin.join_tree,
            num_fidler_vectors: int = 1, num_clusters: int = 2) -> List[Set[int]]:
    g: nx.Graph = nx.from_numpy_matrix(matrix)
    root = Node("root", cluster=set(g.nodes))
    # subgraph_list contains all nodes with size > cluster_size
    node_list = [root] if len(root.cluster) > cluster_size else []
    while node_list:
        next_list = []
        for node in node_list:
            clusters = partition(nx.subgraph(g, node.cluster), num_fidler_vectors, num_clusters)
            next_list.extend([Node(str(i), parent=node, cluster=cluster) for i, cluster in enumerate(clusters, start=1)])
            node.cluster.clear()
        next_list = list(filter(lambda n: len(n.cluster) > cluster_size, next_list))
        node_list = next_list
    clusters = [node.cluster for node in root.leaves]
    return clusters if join_function is None else join_function(matrix, root, cluster_size)




