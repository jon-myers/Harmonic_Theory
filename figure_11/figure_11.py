import numpy as np
import os,sys,inspect, abjad
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, get_complement_shell, are_roots, make_shell_plot

points_A = np.array((
(0, 0, 0),
(0, 0, 1),
(0, 1, 1),
(1, 1, 1),
(2, 1, 1),
(3, 1, 1),
(2, 1, 2)
))

points_B = np.array((
(0, 0, 0),
(1, 0, 0),
(2, 0, 0),
(2, 0, 1),
(2, 0, 2),
(2, 1, 2),
(3, 0, 0),
(3, 1, 0),
(3, 1, 1)
))

points_C = np.array((
(0, 0, 0),
(1, 0, 0),
(1, 1, 0),
(1, 1, 1),
(2, 1, 1),
(3, 1, 1),
(2, 1, 2)
))

shell_A = get_complement_shell(points_A)
shell_B = get_complement_shell(points_B)
shell_C = get_complement_shell(points_C)



primes = np.array((2, 3, 5))

make_shell_plot(shell_A, points_A, primes, currentdir + '/A', dot_size=4,
                transparent=True, shell_color='seagreen')

make_shell_plot(shell_B, points_B, primes, currentdir + '/B', dot_size=4,
                transparent=True, shell_color='seagreen')

make_shell_plot(shell_C, points_C, primes, currentdir + '/C', dot_size=4,
                transparent=True, shell_color='seagreen')
