import numpy as np
import os,sys,inspect, json
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios, are_roots
from utils import cast_to_ordinal, are_root_breakpoints, are_extremities, are_extremity_breakpoints

primes = np.array((3, 5, 7))
          
with open('chords/chords5.json') as json_file:
    chords = json.load(json_file)
    points = [np.array(i['points']) for i in chords]
    
    num_extremities = np.array([np.count_nonzero(are_extremities(chord)) for chord in points])
    num_extremity_breakpoints = np.array([np.count_nonzero(are_extremity_breakpoints(chord)) for chord in points])

    # mask = np.logical_and(num_extremity_breakpoints == 1, num_extremities == 4)
    indexes = np.arange(len(points))[num_extremities == 2]
    choices = np.random.choice(indexes)
    A = cast_to_ordinal(points[choices])
    A_colors = np.repeat(0, len(A))
    A_colors = np.where(are_extremities(A), 1, A_colors)
    A_colors = np.where(are_extremity_breakpoints(A), 2, A_colors)
    A_colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in A_colors]
    make_plot(A, primes, currentdir+'/A', dot_size=2, colors=A_colors,
              ratios=False, origin=False, range_override=[-1, 3], connect_color='black', connect_size=1,
              legend=False, origin_range=[-1, 3], transparent=True)    
    
    mask = np.logical_and(num_extremity_breakpoints == 1, num_extremities == 3)
    indexes = np.arange(len(points))[mask]
    choices = np.random.choice(indexes)
    B = cast_to_ordinal(points[choices])
    B_colors = np.repeat(0, len(B))
    B_colors = np.where(are_extremities(B), 1, B_colors)
    B_colors = np.where(are_extremity_breakpoints(B), 2, B_colors)
    B_colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in B_colors]
    make_plot(B, primes, currentdir+'/B', dot_size=2, colors=B_colors,
              ratios=False, origin=False, range_override=[-1, 3], connect_color='black', connect_size=1,
              legend=False, origin_range=[-1, 3], transparent=True)
    
    mask = np.logical_and(num_extremity_breakpoints == 2, num_extremities == 3)
    indexes = np.arange(len(points))[mask]
    choices = np.random.choice(indexes)
    C = cast_to_ordinal(points[choices])
    C_colors = np.repeat(0, len(C))
    C_colors = np.where(are_extremities(C), 1, C_colors)
    C_colors = np.where(are_extremity_breakpoints(C), 2, C_colors)
    C_colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in C_colors]
    make_plot(C, primes, currentdir+'/C', dot_size=2, colors=C_colors,
              ratios=False, origin=False, range_override=[-1, 3], connect_color='black', connect_size=1,
              legend=False, origin_range=[-1, 3], transparent=True)
    
    mask = np.logical_and(num_extremity_breakpoints == 3, num_extremities == 3)
    indexes = np.arange(len(points))[mask]
    choices = np.random.choice(indexes)
    D = cast_to_ordinal(points[choices])
    D_colors = np.repeat(0, len(D))
    D_colors = np.where(are_extremities(D), 1, D_colors)
    D_colors = np.where(are_extremity_breakpoints(D), 2, D_colors)
    D_colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in D_colors]
    make_plot(D, primes, currentdir+'/D', dot_size=2, colors=D_colors,
              ratios=False, origin=False, range_override=[-1, 3], connect_color='black', connect_size=1,
              legend=False, origin_range=[-1, 3], transparent=True)
    
    
