import networkx as nx
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd
from typing import List, Set, Tuple
import numpy as np

import scipy.cluster as sc

ZERO = 0.00001

def is_graph_connected(laplacian_eig_values: np.ndarray) -> bool:
    # graph is not connected if the second lowest eigenvalue of the laplacian matrix is 0
    zero_eig_value_qty = np.sum(np.abs(laplacian_eig_values) < ZERO)
    return True if zero_eig_value_qty < 2 else False


def cluster_connected_components(eig_values: np.ndarray, eig_vectors: np.ndarray) -> List[Set[int]]:
    # eig_values and eig_vectors of the laplacian matrix of a disconnected graph
    zero_eig_value_indexes = np.flatnonzero(np.abs(eig_values) < ZERO)
    zero_eig_value_vectors = eig_vectors[:,zero_eig_value_indexes].transpose()
    # clusters are made of nodes with nonzero values in the eig_vectors which eig_values are 0
    clusters = [set(np.flatnonzero(np.abs(v) > ZERO)) for v in zero_eig_value_vectors]
    return clusters


def fidler_vector_connected_partition(fidler_vectors: np.ndarray, num_clusters: int, labels: np.ndarray) -> List[Set[int]]:
    #fidler_vectors are the eigenvectors of the n smallest nonzero eigenvalues
    condensed_matrix_distance = ssd.pdist(fidler_vectors)
    linkage_matrix = shc.linkage(condensed_matrix_distance, 'single')
    cluster_assignment = shc.fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    #cluster_assignment starts labeling from 1
    non_labeled_clusters = [np.flatnonzero(cluster_assignment == i) for i in np.arange(num_clusters)+1]
    clusters = [set(labels[cluster]) for cluster in non_labeled_clusters]
    return clusters


def fidler_vector_partition(graph: nx.Graph, num_fidler_vectors: int = 1, num_clusters: int = 2) -> List[Set[int]]:
    laplacian = nx.laplacian_matrix(graph).toarray()
    eig_values, eig_vectors = np.linalg.eig(laplacian)
    if is_graph_connected(eig_values):
        #eigen_vectors (columns) sorted by smallest to biggest eigenvalue, first eigenvalue is 0
        eig_vectors_sorted = eig_vectors[:, eig_values.argsort()]
        fidler_vectors = eig_vectors_sorted[:, 1:num_fidler_vectors+1]
        clusters = fidler_vector_connected_partition(fidler_vectors, num_clusters, np.asarray(graph.nodes))
    else:
        clusters = [set(nx.subgraph(graph, g).nodes) for g in nx.connected_components(graph)]
    return clusters


def fidler_cluster(matrix: np.array, cluster_size: int, num_fidler_vectors: int = 1, num_clusters: int = 2)\
        -> List[Set[int]]:
    graph: nx.Graph = nx.from_numpy_matrix(matrix)
    subgraph_list = [graph]
    while True:
        next_subgraph_list = []
        for subgraph in subgraph_list:
            if subgraph.number_of_nodes() > cluster_size:
                clusters = fidler_vector_partition(subgraph, num_fidler_vectors, num_clusters)
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





