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


def FidlerVectorPartition(G):
    fidler_sub_G_list = [];
    nodes_label_list = list(G.nodes)
    l = nx.laplacian_matrix(G).todense()
    # cheking for if G is connected
    eig_val, eig_vec = np.linalg.eig(l)
    eig_vec_sorted = eig_vec[:, eig_val.argsort()]
    bool_vec = np.abs(eig_val) < 0.00001
    if np.sum(bool_vec)>1: #  if G is not connected, it is partition to its connected compoenets
        """
        for i in range(len(eig_val)):
            if np.abs(eig_val[i]) < 0.00001:
                sub_G_node_idx_list = []
                for j in range(len(eig_vec[:])):
                    if np.abs(eig_vec[j,i]) > 0.00001:
                        sub_G_node_idx_list.append(nodes_label_list[j])
                        
                fidler_sub_G_list.append(nx.subgraph(G,sub_G_node_idx_list))
        """
        fidler_sub_G_list = [nx.subgraph(G, g) for g in nx.connected_components(G)]
                
    else: # if G is connected its rows are projected on Fidler vector and the projction values are divided into two component using mutual proximity criteria    
        fidler_vec = nx.linalg.algebraicconnectivity.fiedler_vector(G) #fidler_vec = eig_vec_sorted[:, 1].transpose()
        l_dot_fidler = fidler_vec*l
        Z = shc.linkage(ssd.pdist(l_dot_fidler.transpose()),'single')
        classInds = shc.fcluster(Z,2,criterion = 'maxclust')
        classA = []
        classB = []
        
        for i in range(0,len(classInds)):
            if classInds[i] == 1:
                classA.append(nodes_label_list[i])
            else:
                classB.append(nodes_label_list[i])
        fidler_sub_G_list.append(nx.subgraph(G,classA))
        fidler_sub_G_list.append(nx.subgraph(G,classB))
        
    return fidler_sub_G_list

def Fidler_Clustering(G):
    fidler_sub_G_list = []
    
    fidler_sub_G_list.append(G)
    running_flag = 1
    while running_flag:
        running_flag = 0
        fidler_sub_G_list_temp = []
        for i in range(0, len(fidler_sub_G_list)):
            if len(fidler_sub_G_list[i].nodes) > sub_graph_node_qty:
                running_flag = 1
                fidler_sub_G_single_partition_list = FidlerVectorPartition(fidler_sub_G_list[i])
                for sub_G in fidler_sub_G_single_partition_list:
                    fidler_sub_G_list_temp.append(sub_G)
            else:
                fidler_sub_G_list_temp.append(fidler_sub_G_list[i])
    
        fidler_sub_G_list = fidler_sub_G_list_temp
        print('Fidler_Work:' + str([g.number_of_nodes() for g in fidler_sub_G_list]))
    return fidler_sub_G_list

plt.close('all')

node_qty = 20
sub_graph_node_qty = 4;

G = nx.gnp_random_graph(node_qty, 0.1)
for u,v in G.edges():
    G[u][v]['weight'] = np.random.randint(1,10)
    #G[u][v]['weight'] = 10*np.random.rand()
# Analysis of the original graph
plt.figure("Graph")
pos = nx.spring_layout(G)
nx.draw_networkx(G,pos)  # networkx draw()
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
plt.draw()  # pyplot draw()

fidler_sub_G_list = Fidler_Clustering(G) # Clustring by iterative binatic partiton using Fidler vector


plt.figure("Clustring by fidler vecotr")
#nx.draw_networkx(G,pos, node_color = 'y')  # networkx draw()
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
for subgraph in fidler_sub_G_list:
    nx.draw_networkx(subgraph,pos,node_color = 'y')
    labels = nx.get_edge_attributes(subgraph,'weight')
    nx.draw_networkx_edge_labels(subgraph,pos,edge_labels=labels)    
            
