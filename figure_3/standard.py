import numpy as np
import os,sys,inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product

points = np.array((
[0, 0, 0],
[-2, 1, 0],
[-1, 1, 1],
[1, 0, -1]
))

draw_points = np.array((
[0, 1, 0],
[-1, 1, 0],
[1, 0, 0],
[0, 0, -1],
[-1, 0, 0],
[-2, 0, 0],
[-2, 1, 1],
[-1, 0, -1],
[-2, 0, -1],
[1, 1, 0],
[0, 1, 1],
[1, 1, 1]
))

primes = np.array((2.0, 3.0, 5.0))

make_plot(points, primes, currentdir +'/standard', draw_points = draw_points)
