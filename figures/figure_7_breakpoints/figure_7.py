import numpy as np
import os,sys,inspect, abjad
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios, are_roots
from utils import cast_to_ordinal, are_root_breakpoints


primes = np.array((2, 3, 5))


A = np.array((
(0, 1, 0),
(1, 1, 0),
(1, 1, 1),
(2, 2, 1),
(2, 1, 1),
(2, 0, 1)
))

B = np.array((
(1, 0, 2),
(2, 0, 2),
(2, 1, 2),
(2, 2, 2),
(2, 1, 1),
(2, 1, 0),
(2, 0, 1),
))

A = cast_to_ordinal(A)
A_colors = np.repeat(0, len(A))
A_colors = np.where(are_roots(A), 1, A_colors)
A_colors = np.where(are_root_breakpoints(A), 2, A_colors)
A_colors = [['black', 'red', 'purple'][i] for i in A_colors]


B = cast_to_ordinal(B)
B_colors = np.repeat(0, len(B))
B_colors = np.where(are_roots(B), 1, B_colors)
B_colors = np.where(are_root_breakpoints(B), 2, B_colors)
B_colors = [['black', 'red', 'purple'][i] for i in B_colors]

make_plot(A, primes, currentdir+'/A', dot_size=2, colors=A_colors,
          ratios=False, origin=True, connect_color='black', connect_size=1,
          legend=False, origin_range=[-1, 3])

make_plot(B, primes, currentdir+'/B', dot_size=2, colors=B_colors,
          ratios=False, origin=True, connect_color='black', connect_size=1,
          legend=False, origin_range=[-1, 3])
