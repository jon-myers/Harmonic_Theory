import numpy as np
import os,sys,inspect, abjad
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, paths_to_point, get_complement_shell

point = np.array((1, 1, 2))
interval = np.array(((0, 0, 0), point))
paths = paths_to_point(point)

draw_points = paths[:,1:-1]
primes = np.array((2, 3, 5))

# A_draw_points = np.array((
# (1, 0, 0),
# (2, 0, 0),
# (3, 0, 0),
# (3, 1, 0),
# (3, 2, 0),
# ))

# paths = paths_to_point(np.array())
for i in range(len(paths)):
    file_path = currentdir + '/' + str(i)
    make_plot(interval, primes, file_path, draw_points=draw_points[i],
              dot_size=8, connect_color='black', legend=False, transparent=True)



shell = get_complement_shell(interval)
# print(shell)
make_plot(shell, primes, currentdir + '/shell', dot_size = 8, 
          connect_color='black', colors=['black']  +
          ['seagreen' for i in range(len(shell)-2)] + ['black'], 
          legend=False, transparent=True)
