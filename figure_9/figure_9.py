import numpy as np
import os,sys,inspect, abjad
import more_itertools, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, get_transposition_shell, are_roots, make_shell_plot

points_A = np.array((
(0, 0, 0),
(0, 1, 0),
(1, 1, 0),
(2, 1, 0),
(1, 1, 1)
))

points_B = np.array((
(0, 0, 0),
(0, 0, 1),
(0, 1, 0),
(1, 1, 0),
(1, 1, 1),
(1, 2, 0),
(2, 1, 0),
))

points_C = np.array((
(0, 0, 0),
(0, 0, 1),
(0, 1, 1),
(1, 1, 1),
(1, 0, 1),
(2, 0, 1)
))

points_D = np.array((
(0, 0, 0),
(0, 1, 0),
(1, 1, 0),
(2, 1, 0),
(1, 1, 1),
(0, 0, 1),
(0, 1, 1)
))

shell_A = get_transposition_shell(points_A)
shell_B = get_transposition_shell(points_B)
shell_C = get_transposition_shell(points_C)
shell_D = get_transposition_shell(points_D)

primes = np.array((2, 3, 5))

make_shell_plot(shell_A, points_A, primes, currentdir+ '/A', dot_size=4, transparent=True, shell_color='seagreen')
make_shell_plot(shell_B, points_B, primes, currentdir+ '/B', dot_size=4, transparent=True, shell_color='seagreen')
make_shell_plot(shell_C, points_C, primes, currentdir+ '/C', dot_size=4, transparent=True, shell_color='seagreen')
make_shell_plot(shell_D, points_D, primes, currentdir+ '/D', dot_size=4, transparent=True, shell_color='seagreen')

#
# colors = [choices[int(i)] for i in are_roots(points)]
# make_plot(points, primes, currentdir + '/branch', colors=colors, legend=False,
#           connect_color='black', dot_size=3)
#
# colors = [choices[int(i)] for i in are_roots(shell)]
# make_plot(shell, primes, currentdir + '/shell', colors = colors, legend=False,
#           connect_color='black', dot_size=3)
