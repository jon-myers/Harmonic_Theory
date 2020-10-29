import numpy as np
import os,sys,inspect, abjad
import more_itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot

points = np.array((
(0, 0, 0),
(1, 0, 0),
(2, 0, 0),
(1, 1, 0),
(1, 1, 1)
))

primes = np.array((2, 3, 5))
colors = ['red', 'black', 'black', 'black', 'black']

A = points
B = points[:-1]
C = points[:-2]
D = points[:-3]
e = np.array((0, 1, 3, 4), dtype='int')
E = points[e]
f = np.array((0, 1, 3), dtype='int')
F = points[f]
G = points[1:]
make_plot(A, primes, currentdir + '/A', ratios=False, origin=True, colors=colors)
make_plot(B, primes, currentdir + '/B', ratios=False, origin=True, colors=colors[:-1])
make_plot(C, primes, currentdir + '/C', ratios=False, origin=True, colors=colors[:-2])
make_plot(D, primes, currentdir + '/D', ratios=False, origin=True, colors=colors[:-3])
make_plot(E, primes, currentdir + '/E', ratios=False, origin=True, colors=[colors[i] for i in e])
make_plot(F, primes, currentdir + '/F', ratios=False, origin=True, colors=[colors[i] for i in f])
make_plot(G, primes, currentdir + '/G', ratios=False, origin=True, colors=colors[:-1])
make_plot(H, primes, )
