import networkx as nx
import random
from networkx.drawing.nx_agraph import graphviz_layout
from utils import are_extremities, are_extremity_breakpoints, are_roots, are_root_breakpoints
import numpy as np
import os,sys,inspect, json
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios, are_roots
from utils import cast_to_ordinal, are_root_breakpoints, are_extremities, are_extremity_breakpoints

    
from utils import create_tree_edges, plot_tree  
# points = np.array((
# (1, 0, 0),
# (0, 1, 0),
# (1, 1, 0),
# (2, 1, 0),
# (1, 2, 0)
# ))

# plot_tree(points, 'test')
primes = np.array((3, 5, 7))
# edges = create_tree_edges(points)
with open('chords/chords5.json') as json_file:
    chords = json.load(json_file)
    points = [np.array(i['points']) for i in chords]
    
    num_extremities = np.array([np.count_nonzero(are_extremities(chord)) for chord in points])
    num_extremity_breakpoints = np.array([np.count_nonzero(are_extremity_breakpoints(chord)) for chord in points])
    
    mask = np.logical_and(num_extremity_breakpoints == 3, num_extremities == 3)
    indexes = np.arange(len(points))[mask]
    choices = np.random.choice(indexes)
    A = cast_to_ordinal(points[choices])
    A_colors = np.repeat(0, len(A))
    A_colors = np.where(are_extremities(A), 1, A_colors)
    A_colors = np.where(are_extremity_breakpoints(A), 2, A_colors)
    A_colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in A_colors]
    make_plot(A, primes, currentdir+'/A', dot_size=2, colors=A_colors,
              ratios=False, origin=False, range_override=[-1, 3], connect_color='black', connect_size=1,
              legend=False, origin_range=[-1, 3], transparent=True) 
    plot_tree(A, 'A_')

# import matplotlib.pyplot as plt
# import networkx as nx
# import itertools
# G=nx.MultiDiGraph()
# G.add_edges_from(edges)
# edge_order = []
# for i in itertools.chain.from_iterable(edges):
#     if i not in edge_order: edge_order.append(i)
# edge_order = np.array(edge_order)
# colors = np.repeat(0, len(points))
# colors = np.where(are_extremities(points), 1, colors)
# colors = np.where(are_extremity_breakpoints(points), 2, colors)
# colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in colors]
# colors = [colors[i] for i in edge_order]
# 
# pos=graphviz_layout(G, prog='dot')
# nx.draw(G, pos, with_labels=False, arrows=True, node_color=colors)
# plt.savefig('test.png')
# nx.nx_agraph.write_dot(G,'test.dot')
# plt.savefig('hierarchy.png')
