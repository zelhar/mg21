#%%
import toolz
from toolz.curried import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

plt.ion()

#%%
# inline magic
#%pylab inline

#plot([1,2,3], [2,4,9])

print("lalala\n")

plt.bar([1,2,3,4], [10, 20, 5, 2])

plt.cla()


plt.plot([1,2,3], [2,4,9])
plt.cla()


#%%


G = nx.DiGraph()
G.add_nodes_from(["a", "b", "c"])
G.add_node("d")
G.add_edge("a", "b", weight=3)
G.add_weighted_edges_from([("b", "c", 2), ("c", "d", 5), ("d", "c", 2)])

adj = nx.to_numpy_array(G)
adj


nx.draw_spring(G)


plt.show()

plt.close()

#%%

#%%

# x = np.arange(5)
# y = x**2
# y
# plt.scatter(x,y)

G = nx.convert_node_labels_to_integers(G, label_attribute="name")

mylabels = G.nodes.data('name')
mylabels = dict(mylabels)
mylabels

nx.draw_spring(G, labels=mylabels, node_size=[100,200,300,400],
        node_color=[0,1,2,3])

plt.close()


G.nodes()
G.nodes[0]["name"]
[G.nodes[i]["name"] for i in G.nodes()]

nx.draw_spring(G, with_labels=True)

nx.draw_spring(G, with_labels=True, labels=mylabels)

#%%


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

#%%

l = [("a", "b"), ("a", "a"), ("b", "a")]

ll = list(map(lambda x: list(x) + ['1'], l))

l = [("a", "b", 2), ("a", "a", 3), ("b", "a", 3), ("a", "b", 5)]

g = nx.MultiDiGraph()
# g = nx.from_edgelist(l)

g.add_weighted_edges_from(l)

nx.draw_spring(g)

nx.to_numpy_array(g)

p = "./data_netcore/CPDB_high_confidence.txt"
X = fromSavedEdgeList(p, digraph=False)

X.edges.data('weight')

A = nx.to_numpy_array(X)
At = A.T / A.sum(axis=1)
At

x = np.array([[1,1], [0,1]])
x
y = np.linalg.inv(x)
y
np.dot(x,y)

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



x = diffusionMatrixG(X)
x

y = nx.to_numpy_array(X)
y
y = y.T / y.sum(axis=1)
y
y = diffusionMatrix(y)
y


t = RWRG(X)
t
tt = np.dot(x, np.ones(3) / 3)
tt

## Coreness

G = nx.karate_club_graph()

club = ['red' if G.nodes[i]['club']=='Mr. Hi' else 'yellow' for i in G.nodes()]
club

nx.draw_spring(G, with_labels=True, node_color=club)

nx.draw_spring(nx.k_shell(G), with_labels=True)

nx.draw_spring(nx.k_core(G,2), with_labels=True)

nx.draw_spring(nx.k_core(G,3), with_labels=True)

nx.draw_spring(nx.k_core(G,4), with_labels=True)

nx.draw_spring(nx.k_core(G), with_labels=True)

plt.close()

coreness = nx.core_number(G)
coreness = np.array([coreness[k] for k in range(len(coreness))])
coreness

A = nx.to_numpy_array(G)
A = A * coreness
A
T = A.T / A.sum(axis=1)
T.sum(axis=0)

a = coreTransitionMatrixG(G)
a
a == T

b = transitionMatrixG(G)
b.sum(axis=0)
a.sum(axis=0)


x = np.arange(16).reshape((4,4))
y = np.array([1,-2,3,-4])


n = len(a)

plt.cla()
nx.draw_spring(G, with_labels=True, node_color=club, node_size=np.dot(a,
    np.ones(n)*100))

plt.cla()
nx.draw_spring(G, with_labels=True, node_color=club, node_size=np.dot(b,
    np.ones(n)*100))

plt.cla()
nx.draw_spring(G, with_labels=True, node_color=club, node_size=100*coreness)


plt.cla()
nx.draw_spring(G, with_labels=True, node_color=club, node_size=100*coreness,
        labels=nx.core_number(G))





K = diffusionMatrixG(G, coreness=False)
R = diffusionMatrixG(G, coreness=True)

G.nodes.data()
G.nodes.data('club')
clubdict = dict( G.nodes.data('club'))

plt.cla()
nx.draw_spring(G, with_labels=True, node_color=club, node_size=10*coreness,
        labels=clubdict)

r = np.dot(R, np.ones(len(R)))
r.sum()
s = np.dot(K, np.ones(len(K)))
s.sum()


plt.cla()
nx.draw_spring(G, with_labels=True, node_color=club, node_size=100*r,
        labels=clubdict)

plt.cla()
nx.draw_spring(G, with_labels=True, node_color=club, node_size=100*s,
        labels=clubdict)











