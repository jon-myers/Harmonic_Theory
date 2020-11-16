from utils import make_plot, are_roots, cast_to_ordinal
import numpy as np

primes = np.array((2, 3, 5))
points = np.array((
    (1, 0, 0), 
    (1, 1, 0), 
    (1, 2, 0), 
    (0, 2, 0), 
    (1, 1, 1), 
    (0, 1, 1), 
    (0, 0, 2),
    (0, 1, 2),
    (0, 1, 1)
))
points = cast_to_ordinal(points)

choices = ['red', 'black']
colors = [choices[not i] for i in are_roots(points)]
make_plot(points, primes, 'scratch', dot_size=2, colors=colors, origin=True, 
          legend=False, connect_color='black', connect_size=1, 
          origin_range=[-1, 3])
