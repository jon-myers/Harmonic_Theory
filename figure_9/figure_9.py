import numpy as np
import os,sys,inspect, abjad
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, get_transposition_shell

points = np.array((
(0, 0, 0),
(0, 1, 0),
(1, 1, 0),
(2, 1, 0),
(3, 1, 0),
(1, 1, 1)
))

primes = np.array((2, 3, 5))

make_plot(points, primes, currentdir + '/branch')
