import networkx as nx
import scipy.cluster as sc
import scipy.spatial.distance as ssd
from typing import List, Set, Tuple
import numpy as np
import sklearn.cluster as skc


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


def cluster(matrix: np.ndarray, cluster_size: int,
                   num_fidler_vectors: int = 1, num_clusters: int = 2) -> List[Set[int]]:
    graph: nx.Graph = nx.from_numpy_matrix(matrix)
    subgraph_list = [graph]
    while True:
        next_subgraph_list = []
        for subgraph in subgraph_list:
            if subgraph.number_of_nodes() > cluster_size:
                clusters = partition(subgraph, num_fidler_vectors, num_clusters)
                for cluster in clusters:
                    next_subgraph_list.append(nx.subgraph(graph, cluster))
            else:
                next_subgraph_list.append(subgraph)
        subgraph_list = next_subgraph_list
        number_of_nodes = np.asarray([g.number_of_nodes() for g in subgraph_list])
        if np.sum(number_of_nodes > cluster_size) == 0:
            break
    clusters = [set(subgraph.nodes) for subgraph in subgraph_list]
    return clusters





