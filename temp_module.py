import toolz
from toolz.curried import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


plt.ion()

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

# x = np.arange(5)
# y = x**2
# y
# plt.scatter(x,y)

G = nx.convert_node_labels_to_integers(G, label_attribute="name")

G.nodes()
G.nodes[0]["name"]
[G.nodes[i]["name"] for i in G.nodes()]


def fromSavedEdgeList(pathToEdgeList):
    """Reads an edge list from a file.
    Returns a networkx multi digraph.
    The nodes labels will be integers 0,1..
    and their names in the list is kept as a 'name'
    attribute.
    """
    f = open(pathToEdgeList, "r")
    es = f.readlines()
    f.close()
    es = list(map(str.split, es))
    g = nx.MultiDiGraph()
    if len(es[0]) == 2:
        g.add_edges_from(es)
    elif len(es[0]) == 3:
        g.add_weighted_edges_from(es)
    g = nx.convert_node_labels_to_integers(g, label_attribute="name")
    return g


l = [("a", "b"), ("a", "a"), ("b", "a")]

l = [("a", "b", 2), ("a", "a", 3), ("b", "a", 3), ("a", "b", 5)]

g = nx.MultiDiGraph()
# g = nx.from_edgelist(l)

g.add_weighted_edges_from(l)

nx.draw_spring(g)

nx.to_numpy_array(g)

p = "./data_netcore/CPDB_high_confidence.txt"
X = fromSavedEdgeList(p)

s = X[0]
s
str.split(s)

list(map(str.split, X))

list(map(lambda x: x + 1, range(10)))
