import numpy as np
import os,sys,inspect, abjad
import more_itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios

A = np.array((
(0, 0, 0),
(1, 0, 0),
(1, 1, 0),
(1, 1, 1),
(2, 1, 1)
))

colors = ['red', 'black', 'black', 'black', 'black', 'black']
make_plot(A, np.array((2, 3, 5)), currentdir + '/A', ratios=False, dot_size=3, colors=colors, origin=True)

B = np.array((
(1, 0, 0),
(0, 1, 0),
(1, 1, 0),
(1, 1, 1),
(2, 1, 1)
))

colors = ['red', 'red', 'black', 'black', 'black', 'black']
make_plot(B, np.array((2, 3, 5)), currentdir + '/B', ratios=False, dot_size=3, colors=colors, origin=True)


C = np.array((
    (2, 0, 1),
    (2, 1, 1),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (1, 1, 2),
    (1, 2, 2)
))

colors = ['red', 'black', 'red', 'black', 'red', 'black', 'black']
make_plot(C, np.array((2, 3, 5)), currentdir + '/C', ratios=False, dot_size=3, colors=colors, origin=True)
