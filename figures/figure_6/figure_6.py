import numpy as np
import os,sys,inspect, abjad
import more_itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product, get_ratios

negative_points = np.array((
[-1, 0, 0],
[-1, 0, 1],
[-1, 0, -1],
[-1, 0, -2],
[-1, -1, 0],
[-2, -1, 0],
[0, -1, 0]
))
positive_points = negative_points + np.array((2, 1, 2))


# transposed_points = np.transpose(positive_points, axes=(2, 0, 1))
transposed_points = np.roll(positive_points, 1, axis=1)
print(negative_points, '\n\n', positive_points, '\n\n', transposed_points)
primes = np.array((3.0, 5.0, 7.0))
octaves = np.array((0, 0, 0))



make_plot(negative_points, primes, currentdir + '/negative', octaves, dot_size=2, 
          ratios=False, origin=True)

make_plot(positive_points, primes, currentdir + '/positive', octaves, dot_size=2, 
          ratios=False, origin=True)

make_plot(transposed_points, primes, currentdir + '/transposed', octaves, dot_size=2, 
          ratios=False, origin=True)
