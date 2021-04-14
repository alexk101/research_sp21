import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import math
import copy
from emd import emd
import networkx as nx


def graphify(partition, name):
    p1_n = 0
    p1_temp = {}
    for k, v in partition.items():
        p1_n = p1_n + len(v)
        p1_temp[name+"_"+str(k)] = v
    emd_X = []
    emd_X_weight = []
    for k, v in p1_temp.items():
        emd_X.append(k)
        emd_X_weight.append(len(v)*1.0/p1_n)
    return (emd_X, emd_X_weight)

def k_consensus(partition_list_in, threshold):
    nodes = []
    weights = []

    for part in partition_list_in:
        node, weight = graphify(part)
        nodes.append(node)
        weights.append(weight)

    d = []
    for x in range(len(nodes)):
        for y in range(
    for kx, vx in p1_temp.items():
        arr = []
        for ky, vy in p2_temp.items():
            arr.append(dist(set(vx), set(vy))*1.0)
        d.append(arr)
    ad, f = emd(emd_X, emd_Y, X_weights=np.array(emd_X_weight), Y_weights=np.array(emd_Y_weight), distance='precomputed',
        D=np.array(d), return_flows=True)

    G=nx.Graph()
    for i in range(0,len(emd_X)):
        G.add_node(emd_X[i], color="red", weight=emd_X_weight[i])
    for i in range(0,len(emd_Y)):
        G.add_node(emd_Y[i], color="blue", weight=emd_Y_weight[i])
