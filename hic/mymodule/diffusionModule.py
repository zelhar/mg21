import toolz
from toolz.curried import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def fromSavedEdgeList(pathToEdgeList, digraph=True):
    """Reads an edge list from a file.
    input pathToEdgeList: path for the edge list file.
    input digraph: bool. If False the graph will be undirected.
    output g: a networkx multi digraph or an undirected multi graph.
    The nodes labels will be integers 0,1..
    and their names in the list is kept as a 'name'
    node attribute.
    If the edge list is of doubles: edge weights will all be 1.
    If the edge list is of triplets: the third element will be the edge weight.
    """
    f = open(pathToEdgeList, "r")
    es = f.readlines()
    f.close()
    es = list(map(str.split, es))
    if digraph == True:
        g = nx.MultiDiGraph()
    else:
        g = nx.MultiGraph()
    if len(es[0]) == 2:
        es = list(map(lambda x: x + [1], es))
        #print(es)
        #g.add_edges_from(es)
        g.add_weighted_edges_from(es)
    elif len(es[0]) == 3:
        es = list(map(lambda x: x[0:2] + [int(x[2])], es))
        g.add_weighted_edges_from(es)
    g = nx.convert_node_labels_to_integers(g, label_attribute="name")
    return g

def transitionMatrixG(G):
    """input G: a graph.
    output T: the transition matrix of G,
    column normalized.
    """
    A = nx.to_numpy_array(G)
    T = A.T / A.sum(axis=1)
    return T

def coreTransitionMatrixG(G):
    """Similar to transitionMatrixG but 
    Returns the core normalized transition matrix of G.
    The cloumns are normalized.
    """
    A = nx.to_numpy_array(G)
    coreness = nx.core_number(G)
    coreness = np.array([coreness[k] for k in range(len(coreness))])
    A = A * coreness
    T = A.T / A.sum(axis=1)
    #T.sum(axis=0)
    return T

def diffusionMatrix(T, alpha=0.2):
    """
    input T: a transition matrix (column normalized).
    input alpha: a the restart probability.
    Output K: the diffusion matrix, which is
    K = a [I - (1-a)T]^(-1)
    """
    n = T.shape[0]
    I = np.identity(n)
    K = I - (1 - alpha)*T
    K = alpha * np.linalg.inv(K)
    return K

def diffusionMatrixG(G, alpha=0.2, coreness=False):
    """
    input G: a networkz graph.
    input alpha: the restart parameter.
    input bool coreness: If True, the normalization uses core number rather
    than the standard adjacency matrix.
    Output K: the diffusion matrix, which is
    K = a [I - (1-a)T]^(-1)
    """
    #A = nx.to_numpy_array(G)
    #T = A.T / A.sum(axis=1)
    if coreness:
        T = coreTransitionMatrixG(G)
    else:
        T = transitionMatrixG(G)
    n = T.shape[0]
    I = np.identity(n)
    K = I - (1 - alpha)*T
    K = alpha * np.linalg.inv(K)
    return K


def RWR(T, alpha=0.2, q=1, epsilon=1e-6, maxiter=10**6):
    """Calculates the stationary distribution of a RWR process
    using the power method.
    input T: a transition matrix (column normalized).
    input alpha: restart probability.
    input q: restart distribution. If none is provided the uniform distribution
    is used (pageRank).
    input epsilon: the stop condition for the convergence.
    input maxiter: maximum number of iterations if convergence isn't reached.
    output p: the stationary distribution
    """
    n = T.shape[0]
    if q==1:
        q = 1/n * np.ones(n)
    x = q
    y = alpha * q + (1 - alpha) * np.dot(T, x)
    #while np.linalg.norm((x-y)) > epsilon:
    for _ in range(maxiter):
        x = y
        y = alpha * q + (1 - alpha) * np.dot(T, x)
        if np.linalg.norm((x-y)) < epsilon:
            break
    return y


def RWRG(G, alpha=0.2, q=1, epsilon=1e-6, maxiter=10**6):
    """Calculates the stationary distribution of a RWR process
    using the power method.
    input G: a networkx graph.
    input alpha: restart probability.
    input q: restart distribution. If none is provided the uniform distribution
    is used (pageRank).
    input epsilon: the stop condition for the convergence.
    input maxiter: maximum number of iterations if convergence isn't reached.
    output p: the stationary distribution
    """
    A = nx.to_numpy_array(G)
    T = A.T / A.sum(axis=1)
    n = T.shape[0]
    if q==1:
        q = 1/n * np.ones(n)
    x = q
    y = alpha * q + (1 - alpha) * np.dot(T, x)
    #while np.linalg.norm((x-y)) > epsilon:
    for _ in range(maxiter):
        x = y
        y = alpha * q + (1 - alpha) * np.dot(T, x)
        if np.linalg.norm((x-y)) < epsilon:
            break
    return y

