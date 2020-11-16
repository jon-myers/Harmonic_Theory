import numpy as np
import os,sys,inspect, abjad
import more_itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, get_transpositions, are_roots

A = np.array((
(0, 0, 0),
(1, 0, 0),
(2, 0, 0),
(2, 1, 0)
))

B = np.array((
(0, 0, 0),
(1, 0, 0),
(2, 0, 0),
(2, 1, 0),
(2, 0, 1)
))

C = np.array((
(0, 0, 0),
(1, 0, 0),
(1, 1, 0),
(1, 1, 1),
(0, 1, 0),
(0, 0, 1),
(1, 0, 1),
(0, 1, 1)
))

primes = np.array((2, 3, 5))

for i, transposition in enumerate(get_transpositions(A)):
    choices = ['black', 'red']
    colors = [choices[int(i)] for i in are_roots(transposition)]
    make_plot(transposition, primes, currentdir + '/A_' + str(i), dot_size=3,
              legend=False, ratios=False, origin=False, transparent=True,
              connect_color='black', colors=colors, range_override=[0, 2])


for i, transposition in enumerate(get_transpositions(B)):
    choices = ['black', 'red']
    colors = [choices[int(i)] for i in are_roots(transposition)]
    make_plot(transposition, primes, currentdir + '/B_' + str(i), dot_size=3,
              legend=False, ratios=False, origin=False, transparent=True,
              connect_color='black', colors=colors, range_override=[0, 2])

for i, transposition in enumerate(get_transpositions(C)):
    choices = ['black', 'red']
    colors = [choices[int(i)] for i in are_roots(transposition)]
    make_plot(transposition, primes, currentdir + '/C_' + str(i), dot_size=3,
              legend=False, ratios=False, origin=False, transparent=True,
              connect_color='black', colors=colors, range_override=[0, 2])
