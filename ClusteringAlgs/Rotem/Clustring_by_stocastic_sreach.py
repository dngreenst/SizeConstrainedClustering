# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import networkx as nx
#import scipy.cluster.hierarchy as shc
#import scipy.spatial.distance as ssd
import mlrose
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

def FisableSolution(max_num_of_nodes_in_cluster, cluster_list): 
    for i in range(len(cluster_list)):
        if len(cluster_list[i]) > max_num_of_nodes_in_cluster:
            temp_class = []
            for j in range(max_num_of_nodes_in_cluster):
                temp_class.append(cluster_list[i][j])
            cluster_list[i] = temp_class
            
    for cluster_ind1 in range(len(cluster_list)):
        for node in cluster_list[cluster_ind1]:
            for cluster_ind2 in range(cluster_ind1+1,len(cluster_list)):
                if node in cluster_list[cluster_ind2]:
                    cluster_list[cluster_ind2].remove(node)
    return cluster_list
        
def UtilityFunc(adj_mat, cluster_list, max_num_of_nodes_in_cluster):
    cluster_list = FisableSolution(max_num_of_nodes_in_cluster, cluster_list)
    utility = 0
    for cluster_node_list in cluster_list:
        for i in range(len(cluster_node_list)):
            for j in range(i+1,len(cluster_node_list)):
                utility = utility + adj_mat[cluster_node_list[i],cluster_node_list[j]]
    return utility
    
def ConvertClusterVectorToList(cluser_vector):
    cluser_list = [ [] for i in range(len(cluser_vector)) ]
    for node in range(len(cluser_vector)):
        cluser_list[int(cluser_vector[node])].append(node)
    while [] in cluser_list:
        cluser_list.remove([])
    return cluser_list

def ConvertListToClusterVector(cluster_list, node_qty):
    cluser_vector = -1*np.ones(node_qty)
    [int(i) for i in cluser_vector]
    for cluster_ind in range(len(cluster_list)):
        for node in cluster_list[cluster_ind]:
            cluser_vector[node] = cluster_ind
    return cluser_vector

def UtilityFunc_main(cluser_vector, adj_mat, max_num_of_nodes_in_cluster):
    
    cluster_list = ConvertClusterVectorToList(cluser_vector)
    return UtilityFunc(adj_mat, cluster_list, max_num_of_nodes_in_cluster)

def GreadySearch1(adj_mat, max_num_of_nodes_in_cluster):
    edge_list =[]
    cluster_list = []
    for i in range(adj_mat.shape[0]):
        for j in range(i+1,adj_mat.shape[1]):
            edge_list.append([adj_mat[i,j],[i,j]])
    edge_list = sorted(edge_list, key=lambda edge_list_entry: edge_list_entry[0])
    edge_list = edge_list[::-1]
    cluser_vector = [ [] for i in range(len(adj_mat)) ]
    for val, edge in edge_list:
        
        if cluser_vector[edge[0]] == [] and cluser_vector[edge[1]] == []:
            cluser_vector[edge[0]] = len(cluster_list)
            cluser_vector[edge[1]] = len(cluster_list)
            cluster_list.append(list([edge[0],edge[1]]))
            
        elif cluser_vector[edge[0]] != [] and cluser_vector[edge[1]] == []:
            if len(cluster_list[cluser_vector[edge[0]]]) < max_num_of_nodes_in_cluster:
                cluser_vector[edge[1]] = cluser_vector[edge[0]]
                cluster_list[cluser_vector[edge[0]]].append(edge[1])
            else:
                cluser_vector[edge[1]] = len(cluster_list)
                cluster_list.append(list([edge[1]]))
        
        elif cluser_vector[edge[1]] != [] and cluser_vector[edge[0]] == []:
            if len(cluster_list[cluser_vector[edge[1]]]) < max_num_of_nodes_in_cluster:
                cluser_vector[edge[0]] = cluser_vector[edge[1]]
                cluster_list[cluser_vector[edge[1]]].append(edge[0])
            else:
                cluser_vector[edge[0]] = len(cluster_list)
                cluster_list.append(list([edge[0]]))    
        
        elif cluser_vector[edge[1]] != [] and cluser_vector[edge[0]] != []:
            if len(cluster_list[cluser_vector[edge[1]]]) + len(cluster_list[cluser_vector[edge[0]]]) < max_num_of_nodes_in_cluster:
                for node in cluster_list[cluser_vector[edge[1]]]:
                    cluster_list[cluser_vector[edge[0]]].append(node)
                    cluser_vector[node] = cluser_vector[edge[0]]
    return cluser_vector, UtilityFunc(adj_mat, cluster_list, max_num_of_nodes_in_cluster)     
        
def GreadySearch2(adj_mat, max_num_of_nodes_in_cluster):
    N = len(adj_mat)
    open_node_list = list(range(N))
    cluster_list = [];
    while len(open_node_list) > 0:
        Curr_Cluster =[]
        node_ind = np.random.randint(0,len(open_node_list))
        best_node = open_node_list[node_ind]
        while len(open_node_list) > 0 and len(Curr_Cluster) < max_num_of_nodes_in_cluster and best_node != -1:
            open_node_list.remove(best_node)
            Curr_Cluster.append(best_node)
            best_val = 0
            best_node = -1
            for i in open_node_list:
                val = 0
                for j in Curr_Cluster:
                    val = val + adj_mat[i,j]
                if best_val < val:
                    best_val = val
                    best_node = i
        cluster_list.append(Curr_Cluster)
    return ConvertListToClusterVector(cluster_list, N), UtilityFunc(adj_mat, cluster_list, max_num_of_nodes_in_cluster)

def GreadyLoop(adj_mat, max_num_of_nodes_in_cluster, iteration_num):
    BestSolution = []
    BestSolutionVal = -1
    for i in range(iteration_num):
        best_state, best_fitness =  GreadySearch2(adj_mat, max_num_of_nodes_in_cluster)
        if BestSolutionVal < best_fitness:
            BestSolutionVal = best_fitness
            BestSolution    = best_state
    return BestSolution, BestSolutionVal
 
plt.close('all')
Fig_list =[]

node_qty = 50
sub_graph_node_qty = 10   # 


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


adj_mat = nx.adjacency_matrix(G).todense()

kwargs = {'adj_mat': adj_mat , 'max_num_of_nodes_in_cluster' : sub_graph_node_qty}
fitness_cust = mlrose.CustomFitness(UtilityFunc_main, **kwargs)
problem = mlrose.DiscreteOpt(length = node_qty, fitness_fn = fitness_cust, maximize = True, max_val = node_qty)

schedule = mlrose.ExpDecay()
init_state = np.array(range(node_qty))
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10, max_iters = 1000, init_state = init_state)
cluser_list = ConvertClusterVectorToList(best_state)
print('simulated_annealing:')
for cluster in cluser_list:
    print(cluster)
print('The fitness at the best state is: ', best_fitness)

schedule = mlrose.ExpDecay()
init_state, val = GreadySearch2(adj_mat, sub_graph_node_qty)
best_state, best_fitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 10, max_iters = 1000, init_state = init_state)
cluser_list = ConvertClusterVectorToList(best_state)
print('gready2 + simulated_annealing:')
for cluster in cluser_list:
    print(cluster)
print('The fitness at the best state is: ', best_fitness)

best_state, best_fitness = mlrose.genetic_alg(problem, mutation_prob = 0.2, max_attempts = 1000, random_state = 2)
cluser_list = ConvertClusterVectorToList(best_state)
print('genetic_alg:')
for cluster in cluser_list:
    print(cluster)
print('The fitness at the best state is: ', best_fitness)

best_state, best_fitness = mlrose.hill_climb(problem, max_iters = 1000)
cluser_list = ConvertClusterVectorToList(best_state)
print('hill_climb:')
for cluster in cluser_list:
    print(cluster)
print('The fitness at the best state is: ', best_fitness)

best_state, best_fitness = mlrose.random_hill_climb(problem, max_iters = 10000)
cluser_list = ConvertClusterVectorToList(best_state)
print('random_hill_climb:')
for cluster in cluser_list:
    print(cluster)
print('The fitness at the best state is: ', best_fitness)

#best_state, best_fitness =  GreadySearch1(adj_mat, sub_graph_node_qty)

#best_state, best_fitness =  GreadySearch2(adj_mat, sub_graph_node_qty)

best_state, best_fitness =  GreadyLoop(adj_mat, sub_graph_node_qty,10)
cluser_list = ConvertClusterVectorToList(best_state)
print('Gready loop:')
for cluster in cluser_list:
    print(cluster)
print('The fitness at the best state is: ', best_fitness)

