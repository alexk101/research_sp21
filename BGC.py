import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import math
import copy
from emd import emd
import networkx as nx


"""
Jaccard distance between two sets
"""
def jaccard_distance(set_1, set_2):
    isec = set_1.intersection(set_2)
    un = set_1.union(set_2)
    return (1-(len(isec)*1.0)/len(un))

"""
Function that calculates distance between two partitions using EMD
Parameters:
  p1: Partition 1
      A partiton needs to be a dictionary, where keys are strings and values are lists.
      Each key represent the name of a cluster in the partition and corresponding
      list represent points belonging to the cluster.
  p2: Partition 2
  dist: Distance function between individual clusters.
        This distance function should take two sets and return a value between 0 and 1.
        Example: jaccard_distance
"""
def partition_distance(p1, p2, dist):
    p1_n = 0
    p1_temp = {}
    for k, v in p1.items():
        p1_n = p1_n + len(v)
        p1_temp["p1_"+str(k)] = v
    emd_X = []
    emd_X_weight = []
    for k, v in p1_temp.items():
        emd_X.append(k)
        emd_X_weight.append(len(v)*1.0/p1_n)
        
    p2_n = 0
    p2_temp = {}
    for k, v in p2.items():
        p2_n = p2_n + len(v)
        p2_temp["p2_"+str(k)] = v
    emd_Y = []
    emd_Y_weight = []
    for k, v in p2_temp.items():
        emd_Y.append(k)
        emd_Y_weight.append(len(v)*1.0/p2_n)
        
    d = []
    for kx, vx in p1_temp.items():
        arr = []
        for ky, vy in p2_temp.items():
            arr.append(dist(set(vx), set(vy))*1.0)
        d.append(arr)
    a, f = emd(emd_X, emd_Y, X_weights=np.array(emd_X_weight), Y_weights=np.array(emd_Y_weight), distance='precomputed',
        D=np.array(d), return_flows=True)
    return a


"""
Implementation of simple censensus clustering algorithm
This algorithm comes to consensus by using EMD as distance between two input partitions
It used the flow returned by EMD to come to consensus
Input:
    `p1`, `p2` : Two partitions whose consensus partition would be calculated
    `dist`: Distance function between two clusters
    `threshold`: Any edge that has flow less that the threshold would be removed
"""
def basic_consensus_two(p1_i1, p2_i2, dist, threshold):
    p1 = p1_i1[0] # Get the raw partition representation of 1st partition
    i1 = p1_i1[1] # Get the indicator representation of 1st partition
    p2 = p2_i2[0] # Get the raw partition representation of 2nd partition
    i2 = p2_i2[1] # Get the indicator representation of 2nd partition
    
    p1_n = 0 # To keep count of total number of elements in the 1st partition
    p1_temp = {} # To rename each cluster of 1st partition. Temporarily needed for convenince of book keeping. Each cluster is named p1_<cluster_number>
    for k, v in p1.items(): # Populate the temporary data structure
        p1_n = p1_n + len(v)
        p1_temp["p1_"+str(k)] = v # Renaming the key representing cluster
    # Making data structure ready for EMD
    emd_X = []
    emd_X_weight = []
    for k, v in p1_temp.items():
        emd_X.append(k) # Name of each cluster
        emd_X_weight.append(len(v)*1.0/p1_n) # Each cluster is given weight proportional to the number of datapoints in it
        
    p2_n = 0 # Same as 1st partition
    p2_temp = {} # Same as 1st partition
    for k, v in p2.items():
        p2_n = p2_n + len(v)
        p2_temp["p2_"+str(k)] = v # Renaming the key representing cluster
    emd_Y = []
    emd_Y_weight = []
    for k, v in p2_temp.items():
        emd_Y.append(k) # Name of each cluster
        emd_Y_weight.append(len(v)*1.0/p2_n) # Each cluster is given weight proportional to the number of datapoints in it
   
    # Precompute distance between each cluster with given distance method
    d = []
    for kx, vx in p1_temp.items():
        arr = []
        for ky, vy in p2_temp.items():
            arr.append(dist(set(vx), set(vy))*1.0)
        d.append(arr)
    ad, f = emd(emd_X, emd_Y, X_weights=np.array(emd_X_weight), Y_weights=np.array(emd_Y_weight), distance='precomputed',
        D=np.array(d), return_flows=True)
    
    # Create a bipartite graph with edge weights calculated by EMD 
    G=nx.Graph()
    for i in range(0,len(emd_X)):
        G.add_node(emd_X[i], color="red", weight=emd_X_weight[i])
    for i in range(0,len(emd_Y)):
        G.add_node(emd_Y[i], color="blue", weight=emd_Y_weight[i])
    color_map = []
    size_map = []
    for node in G:
        node_split = node.split("_") 
        node_split[1] = int(node_split[1])
        if node_split[0][1] == '1':
            color_map.append('red')
        else:
            color_map.append('blue')
    #G.add_nodes_from(emd_X, color="red")
    #G.add_nodes_from(emd_Y, color="blue")
    for i in range(len(f)):
        for j in range(len(f[i])):
            G.add_edge(emd_X[i], emd_Y[j], weight=f[i][j])

    # Remove edges having flow less than a threshold amount
    edges_to_be_removed = []
    for u,v,a in G.edges(data=True):
        if a["weight"] <= ad*threshold:
            edges_to_be_removed.append((u,v))
        else:
            pass
    G.remove_edges_from(edges_to_be_removed)
    
    # nx.draw_networkx(G, pos=nx.spring_layout(G), node_color=color_map, ax=axs)
    
    ic_list = [] # To store list of indicator vectors
    for component in nx.connected_components(G): # For each connected component of bipartite graph after edge removal
        pc_i = np.zeros((p1_n, 1)) # Each component will be a single cluster. So Initialize an indicator vector representation for that
        for e in component: # For each element of the connected component
            # Each vertex is named in a specific way , so parse that to figure out the next step
            e_split = e.split("_") 
            e_split[1] = int(e_split[1]) # Second token which represents the cluster number of a particular partition
            if e_split[0][1] == '1': # If the element came from the first partition
                pc_i = np.append(pc_i, i1[e_split[1]], axis=1) # Append the corresponding indicator vector
            if e_split[0][1] == '2': # If the element came from the second partition
                pc_i = np.append(pc_i, i2[e_split[1]], axis=1) # Similar to the first condition
        pc_i_sum = np.sum(pc_i, axis=1) # Sum all the indicator vectors together
        pc_i_sum = pc_i_sum.reshape((p1_n, 1))
        ic_list.append(pc_i_sum)
    
    pc_c = [] # Would have as many elements as the number of data points
    pc_i = [] # Would have as many elements as the number of data points
    for i in range(0, p1_n): # Iterate over all datapoints
        max_sum_i = 0
        max_sum_c = 0
        for c in range(0, len(ic_list)): # Iterate over all clusters found by merging
            if ic_list[c][i] > max_sum_i: # Check if datapoint i has highest vote for cluster c
                # Keep track for which cluster a datapoint has highest vote
                max_sum_i = ic_list[c][i]
                max_sum_c = c
        # Save cluster assignments
        pc_c.append(max_sum_c)
        pc_i.append(max_sum_i)
        
    # Prepare consensus partition in the same way as the two input partitions
    pc_labels_set = set(pc_c)
    pc_clusters = {}
    pc_indicator = {}
    # Initialize two dictionaries to store final cluster representation and corresponding indicator vectors
    for e in pc_labels_set:
        pc_clusters[e] = []
        # pc_indicator[e] = np.zeros((len(pc_c),1))
        pc_indicator[e] = np.array(ic_list[e])
    # Populate the dictionaries
    for i in range(len(pc_c)):
        pc_clusters[pc_c[i]].append(i)
        # pc_indicator[pc_c[i]][i] = pc_i[i]
    return (pc_clusters, pc_indicator)

"""
Hierarchical consensus clustering strategy
Parameters: list of partitions

Each partition is expected to be a tuple with two dictionaries
First dictionary is expected to contain raw partition
while the second dictionary is expected to contain indicator vector representation
of the partition.
"""
def basic_consensus(partition_list_in, threshold):
    #fig = plt.figure(figsize=(18, 18))
    #gs = GridSpec(nrows=3, ncols=3)
    #ax = fig.add_subplot(gs[0,0])
    partition_list=copy.deepcopy(partition_list_in)
    while(len(partition_list) > 1): # Continue until there is only one partition left in the list
        min_d = math.inf # Set current minimum distance as infinity
        min_p1 = None
        min_p2 = None
        # Iterate over all pair of elements of partition list
        for i in range(0, len(partition_list)):
            for j in range(i+1, len(partition_list)):
                # Get distance between a pair of partitions
                # Pass only the first element of the tuple which has raw partition representation
                # Indicator representation is not used here. That representation is used when merging
                # Pass Jaccard distance as distance function
                pd = partition_distance(partition_list[i][0], partition_list[j][0], jaccard_distance)
                if pd < min_d:
                    min_d = pd
                    min_p1 = partition_list[i]
                    min_p2 = partition_list[j]
        # Merge two partitions having least distance.
        # Pass whole partition, both raw representation as well as the indicator representation.
        # Cut the edges having flow less than the threshold
        (pc_clusters, pc_indicator) = basic_consensus_two(min_p1, min_p2, jaccard_distance, threshold)
        # Remove the two candidate partitions from the list
        partition_list.remove(min_p1)
        partition_list.remove(min_p2)
        # Add the new merged partition to the list
        partition_list.append( (pc_clusters, pc_indicator) )
    # Return the only remaining partition in the list
    return partition_list[0]

"""
Transform the output from the basic_consensus function into a one dimensional array of values
Parameters: 
    input: The pc_cluster output from the basic consensus function
    size: The number of values contained within the input

The function unpacks the dictionary containing the class values of the output to allow easier
manipulation of the output by other functions
"""
def output_to_array(input, size):
    consensus_y = [0]*size
    for k in input.keys():
        for e in input[k]:
            consensus_y[e] = k+1
            
    uniq_map = {}
    label=0
    for x in range(len(consensus_y)):
        if(not(consensus_y[x] in uniq_map)):
            uniq_map.update({consensus_y[x]:label})
            label=label+1
        
    for x in range(len(consensus_y)):
        consensus_y[x]=uniq_map.get(consensus_y[x])
    
    return np.asarray(consensus_y)
