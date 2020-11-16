import numpy as np
import os,sys,inspect, abjad
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios



A = np.array((
(0, 0, 0),
(0, 1, 0),
(1, 1, 0),
(2, 1, 0),
(1, 1, 1),
(2, 1, 1)
))

c = [np.array(list(i)) for i in itertools.permutations((0, 1, 2))]

primes = np.array((2, 3, 5))
colors = ['red', 'black', 'black', 'black', 'black', 'black']


for i in range(6):
    make_plot(A[:, c[i]], primes, currentdir + '/' + str(i), ratios=False, 
              origin=False, dot_size=4, colors=colors, legend=False, 
              range_override=[-2, 3], transparent=True)
              
