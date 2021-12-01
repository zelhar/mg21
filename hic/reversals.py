import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as op
import pygraphviz as pgv
import toolz
import toyplot as tp

def getLEnd(i,A):
    n = len(A)
    x = str(abs(A[i % n]))
    if A[i % n]<0:
        #return "a"+str(A[i % n])
        return "a"+x
    else:
        #return "b"+str(A[i % n])
        return "b"+x
def getREnd(i,A):
    n = len(A)
    x = str(abs(A[i % n]))
    if A[i % n]<0:
        return "b"+x
        #return "b"+str(A[i % n])
    else:
        #return "a"+str(A[i % n])
        return "a"+x


def createBPG(p,r, plot=True):
    """ parameters p,r: lists of integers which represents a reversal
    cyclic permutation. It will be used to create a breakpoint graph
    against the reference.
    The reference has blue edges. P has green edges. The synteny
    blocks are represented as black edges.
    optional param plot: if True the graph will be plotted.
    """
    p = list(p)
    r = list(r)
    ns = sorted(np.abs(p+r))
    ns = list(np.unique(ns))
    a_nodes = ["a"+str(abs(i)) for i in ns]
    b_nodes = ["b"+str(abs(i)) for i in ns]
    my_nodes = list(toolz.interleave([a_nodes, b_nodes]))
    G = nx.MultiGraph()
    G.add_nodes_from(my_nodes)
    blackedges = list(zip(a_nodes, b_nodes))
    #blocknames = ["s"+str(i) for i in range(len(blackedges))]
    blocknames = ["s"+str(i) for i in ns]
    blockdict = dict(zip(blackedges, blocknames))
    #blueedges = [("b"+str(r[i % len(r)]), "a"+str(r[(i+1) % len(r)])) for i in range(len(r))]
    blueedges = [(getLEnd(i,r), getREnd(i+1,r)) for i in range(len(r))]
    greenedges = [(getLEnd(i,p), getREnd(i+1,p)) for i in range(len(p))]
    greenOnlyedges = [(u,v) for u,v in greenedges if ((u,v) not in blueedges) and ((v,u) not in blueedges)]
    #greenedges = [("b"+str(p[i % len(p)]), "a"+str(p[(i+1) % len(p)])) for i in range(len(p))]
    #G.add_edges_from(zip(a_nodes, b_nodes), color="black", weight=10)
    #G.add_edges_from(zip(b_nodes, a_nodes[1:]+a_nodes[0:1]), color="blue", weight=1)
    pos = nx.circular_layout(G)
    G.add_edges_from(blackedges, color="black", weight=10)
    G.add_edges_from(blueedges, color="blue", weight=1)
    G.add_edges_from(greenedges, color="green", weight=2)
    edgeColors = [d['color'] for (u,v,d) in G.edges(data=True) ]
    nodeColors = len(ns)*["tan", "teal"]
    #nx.draw_spring(G, node_color=nodeColors, edge_color=edgeColors, edge_labels=blockdict)
    nx.draw_networkx_nodes(G,pos, node_shape="s", label= nx.draw_networkx_labels(G,
        pos, font_size=8), node_color=nodeColors, node_size=50,)
    nx.draw_networkx_edges(G, pos,
            edgelist=blackedges,
            edge_color="black",
            label=nx.draw_networkx_edge_labels(G,pos,edge_labels=blockdict), )
    nx.draw_networkx_edges(G, pos,
            edgelist=blueedges,
            edge_color="blue",
            label=nx.draw_networkx_edge_labels(G,pos,edge_labels=blockdict), )
    nx.draw_networkx_edges(G, pos,
            edgelist=greenOnlyedges,
            #edgelist=greenedges,
            edge_color="green",
            label=nx.draw_networkx_edge_labels(G,pos,edge_labels=blockdict), )
    return G

plt.cla()
G=createBPG([1,-3,-2], [1,2,3,4])

createBPG([1,2,3,2,4], [1,2,3,4])

createBPG([1,-2,3,2,4], [1,-2,3,4])


G = nx.dodecahedral_graph()
edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

#plt.ion()

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



### PyGraphViz Tests
G = pgv.AGraph(directed=False, strict=False)

a_nodes = ["a"+str(i) for i in range(10)]
a_nodes
b_nodes = ["b"+str(i) for i in range(10)]
b_nodes
my_nodes = list(toolz.interleave([a_nodes, b_nodes]))
my_nodes

G.add_nodes_from(my_nodes)
G.add_edges_from(zip(a_nodes, b_nodes), color="black", weight=10)
G.add_edges_from(zip(b_nodes, a_nodes[1:]), color="blue", weight=10)
G.add_edge("b9","a0", color="blue")

#01234567890
#0-6-52347890

G.add_edge("b0","b6", color="green")
G.add_edge("a5","a2", color="green")

G.add_edge("a7","b7", color="green")



G.draw("temp.png", prog="circo")

#### more networkx tests
G.clear()
G = nx.MultiGraph()

G.add_nodes_from(my_nodes)
G.add_edges_from(zip(a_nodes, b_nodes), color="black", weight=10)
G.add_edges_from(zip(b_nodes, a_nodes[1:]), color="blue", weight=10)
G.add_edge("b9","a0", color="blue")

G.add_edge("b0","b6", color="green")
G.add_edge("a5","a2", color="green")
G.add_edge("a7","b7", color="green")


edgeColors = [d['color'] for (u,v,d) in G.edges(data=True) ]

plt.cla()

nx.draw_circular(G, with_labels=True, edge_color=edgeColors,
        connectionstyle='arc3,rad=0.2')

pos = nx.circular_layout(G)
pos

plt.cla()

nx.draw_networkx(G, pos, node_size=50, font_size=10, node_shape="s")

nx.draw_networkx_nodes(G,pos, node_shape="s", label= nx.draw_networkx_labels(G,
    pos, font_size=8), node_color=["tan", "teal"]*10, node_size=50,)

blackedges = list(zip(a_nodes, b_nodes))
blackedges
bnames = ["s"+str(i) for i in range(len(blackedges))]

blueEdges = list(zip(b_nodes[-1:]+b_nodes[:-1], a_nodes))
blueEdges

bdict = dict(zip(blackedges, bnames))

nx.draw_networkx_edges(G, pos,
        edgelist=list(zip(a_nodes, b_nodes)),
        edge_color="black",
        label=nx.draw_networkx_edge_labels(G,pos,edge_labels=bdict), )

nx.draw_networkx_edges(G, pos,
        edgelist=blueEdges,
        edge_color="blue",
        )

#nx.draw_networkx_edges(G, pos,
#        edgelist=G.edges(),
#        edge_color=edgeColors)
#

for a,b in zip(a_nodes,b_nodes):
    G.edges[a,b,0]['label'] = "S"+a[1]
    print(G[a][b])

# convert to a graphviz agraph
A = nx.nx_agraph.to_agraph(G)

# write to dot file
A.write("k5_attributes.dot")

A.layout('circo')

A.draw('foo.png')



## toyplot tests
G = tp.graph()
G.add_edges_from?

