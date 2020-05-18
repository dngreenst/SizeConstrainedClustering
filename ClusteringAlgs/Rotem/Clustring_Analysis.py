# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import networkx as nx
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd
import numpy as np

        
def look_for_N_neighbors_in_tree(tree, node_idx, neigbors_N):
    ids = tree.pre_order(lambda x: x.id)
    if any(ids[j] == node_idx for j in range(len(ids))):
        if len(ids) <= neigbors_N+1:
            return ids, tree.get_id()
        else:
            t_l = tree.get_left()
            node_list, nid = look_for_N_neighbors_in_tree(t_l, node_idx, neigbors_N)
            if len(node_list) > 0:
                return node_list, nid
            else:
                t_r = tree.get_right()
                return look_for_N_neighbors_in_tree(t_r, node_idx, neigbors_N)
    else:
        return [],[]

def CreateSubGraphFromRoot1Hope(G,root_node):
    n_list = []
    n_list.append(root_node)
    for u,v in G.edges(root_node):
        if all(n_list[j] != v for j in range(len(n_list))):
            n_list.append(v)   # Add nodes from first hope    
    return G.subgraph(n_list)

def CreateSubGraphFromRoot2Hope(G,root_node):
    n_list = []
    n_list.append(root_node)
    for u,v in G.edges(root_node):
        if all(n_list[j] != v for j in range(len(n_list))):
            n_list.append(v)   # Add nodes from first hope
        for u1, v1 in G.edges(v):
            if all(n_list[j] != v1 for j in range(len(n_list))):
                n_list.append(v1)  # Add nodes from second hope    
    return G.subgraph(n_list)

plt.close('all')
Fig_list =[]

node_qty = 100
sub_graph_node_qty = 7;   # Not relevent with the current implementation
sub_graph_qty  = 5;

G = nx.gnp_random_graph(node_qty, 0.3)
for u,v in G.edges():
    G[u][v]['weight'] = np.random.randint(1,50)
    #G[u][v]['weight'] = 10*np.random.rand()
# Analysis of the original graph
Fig_list.append(plt.figure("Graph"))
pos = nx.spring_layout(G)
nx.draw_networkx(G,pos)  # networkx draw()
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.draw()  # pyplot draw()
L = nx.laplacian_matrix(G)
l = L.todense()

Fig_list.append(plt.figure("Spectral representation"))
pos_spectral=nx.spectral_layout(G)
nx.draw_networkx(G, pos_spectral)
nx.draw_networkx_edge_labels(G,pos_spectral,edge_labels=labels)

Fig_list.append(plt.figure("Dendograms Of Adjacency matrix - dendrogram - metric: 1/(1+p_dist), 'average'"))
A = nx.adjacency_matrix(G)
a = A.todense()
p_dist = ssd.squareform(a)
Z = shc.linkage(1/(1+p_dist), 'single')
dend = shc.dendrogram(Z, labels = list(G.nodes))
# Creating subgraphs

Fig_list.append(plt.figure("Root node dendrograms - metric: 1/(1+p_dist), 'single'"))  

sub_plot_idx = 1
root_node = 0
flag = 1
root_cluster_node_list_temp = []
root_cluster_node_list_temp.append(root_node)
while flag == 1 or len(root_cluster_node_list_temp) > 0:
    current_root_node = root_cluster_node_list_temp[0] #define the node act as the current center
    root_cluster_node_list_temp.remove(current_root_node)
    sub_G = CreateSubGraphFromRoot2Hope(G, current_root_node)
    root_node_idx = list(sub_G.nodes).index(current_root_node)
    a_sub = nx.adjacency_matrix(sub_G).todense()
    p_dist = ssd.squareform(a_sub)
    Z_sub = shc.linkage(1/(1+p_dist), 'single')
    sub_tree = shc.to_tree(Z_sub)
    cluster_node_idx_list, nid = look_for_N_neighbors_in_tree(sub_tree, root_node_idx, sub_graph_node_qty)
  #  display(str(Z_sub[nid]))
    node_label_list_by_idx = list(sub_G.nodes)
    cluster_node_list = []
    for i in cluster_node_idx_list:
        cluster_node_list.append(node_label_list_by_idx[i])
    print(str(current_root_node))
    print(str(cluster_node_idx_list))
    print(str(cluster_node_list))
    
    class_G = G.subgraph(cluster_node_list)
    a_sub_class = nx.adjacency_matrix(class_G).todense()
    p_dist = ssd.squareform(a_sub_class)
    Z_sub_class = shc.linkage(1/(1+p_dist), 'single')
    
#    Fig_list.append(plt.figure("Root node " + str(current_root_node) + " dendrogram - metric: 1/(1+p_dist), 'single'"))  
    if sub_plot_idx <= 4:
        plt.subplot(4,1,sub_plot_idx)
        sub_plot_idx = sub_plot_idx+1
#       dend_sub = shc.dendrogram(Z_sub, labels = list(sub_G.nodes))
        shc.dendrogram(Z_sub_class, labels = list(sub_G.nodes))
    
    
    
    if flag == 1:
        cluster_node_list.remove(root_node)
        root_cluster_node_list_temp = cluster_node_list
        root_cluster_node_list = list([root_node])
        root_cluster_node_list.extend(cluster_node_list)
        flag = 0
        
    
    
