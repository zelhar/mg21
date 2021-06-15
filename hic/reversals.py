import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as op

plt.ion()

G = nx.MultiDiGraph()
G.add_nodes_from("AaBbCcDdEeFfGgHhIiJj", color="yellow")
#G.add_nodes_from("ABCDEFGHIJ", color="yellow")
#G.add_nodes_from("abcdefghij", color="green")
blocks = zip("ABCDEFGHIJ", "abcdefghij")
blocks = list(blocks)
blocks
G.add_edges_from(blocks, color='black', weight=1)



blueEdges = zip("abcdefghij", "BCDEFGHIJA")
blueEdges = list(blueEdges)
blueEdges
G.add_edges_from(blueEdges, color="blue", weight=2)

nodeColorDict = dict(
        G.nodes.data('color'))
nodeColors = nodeColorDict.values()

[(u,v,d) for (u,v,d) in G.edges(data=True) ]
[d['color'] for (u,v,d) in G.edges(data=True) ]

edgeColors = [d['color'] for (u,v,d) in G.edges(data=True) ]


nx.draw_circular(G, with_labels=True, node_color=nodeColorDict.values(),
        edge_color=edgeColors)



plt.cla()

plt.close()

# reverse Ee:
G.add_edges_from(zip("dE","eF"), weight=3, color="green")

edgeColors = [d['color'] for (u,v,d) in G.edges(data=True) ]
