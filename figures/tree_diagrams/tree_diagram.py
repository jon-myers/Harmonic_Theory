import numpy as np
import os,sys,inspect, json
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios, are_roots
from utils import cast_to_ordinal, are_root_breakpoints, are_extremities, are_extremity_breakpoints
from utils import create_tree_edges, plot_tree, plot_basic_hsl

with open('chords/chords5.json') as json_file:
    chords = json.load(json_file)
    points = [np.array(i['points']) for i in chords]
    
    num_roots = np.array([np.count_nonzero(are_roots(chord)) for chord in points])
    num_root_breakpoints = np.array([np.count_nonzero(are_root_breakpoints(chord)) for chord in points])
    num_extremities = np.array([np.count_nonzero(are_extremities(chord)) for chord in points])
    num_extremity_breakpoints = np.array([np.count_nonzero(are_extremity_breakpoints(chord)) for chord in points])

    mask = np.logical_and(num_roots > 1, num_extremities > 1)
    indexes = np.arange(len(points))[mask]
    choice = np.random.choice(indexes)
    A = cast_to_ordinal(points[choice])
    plot_basic_hsl(A, currentdir + '/A', 'root_extremity')
    plot_tree(A, currentdir + '/A_tree', 'root_extremity')
    
    choice = np.random.choice(indexes)
    B = cast_to_ordinal(points[choice])
    plot_basic_hsl(B, currentdir + '/B', 'root_breakpoint')
    plot_tree(B, currentdir + '/B_tree', 'root_breakpoint')
    
    choice = np.random.choice(indexes)
    C = cast_to_ordinal(points[choice])
    plot_basic_hsl(C, currentdir + '/C', 'extremity_breakpoint')
    plot_tree(C, currentdir + '/C_tree', 'extremity_breakpoint')
