import numpy as np
import os,sys,inspect, abjad
import more_itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, sub_branches, are_roots, make_shell_plot, cast_to_ordinal

points = np.array((
(0, 0, 0),
(1, 0, 0),
(2, 0, 0),
(1, 1, 0),
(1, 1, 1)
))

primes = np.array((2, 3, 5))
colors = ['red', 'black', 'black', 'black', 'black']



sub_branches = sub_branches(points)
ordinal_sub_branches = [cast_to_ordinal(i).tobytes() for i in sub_branches]

unique_osbs = list(set(ordinal_sub_branches))
for i, uo in enumerate(unique_osbs):
    out = np.frombuffer(uo, dtype=np.int)
    out = out.reshape((int(len(out)/3), 3))
    unique_osbs[i] = out




#
# z = np.array(unique_osbs[0])
# z = np.frombuffer(z, dtype=np.int)
# z = z.reshape((int(len(z)/3), 3))
# print(z)
for i, sb in enumerate(sub_branches):
    choices = ['black', 'red']
    colors = [choices[int(i)] for i in are_roots(sb)]


    make_shell_plot(points, sb, primes, currentdir + '/' + str(i), ratios=False,
                    origin=False, colors=colors, dot_size=2, legend=False,
                    point_color='black', shell_color='grey',
                    range_override=[-1, 3], transparent=True)


for i, sb in enumerate(unique_osbs):
    choices = ['black', 'red']
    colors = [choices[int(i)] for i in are_roots(sb)]

    make_plot(sb, primes, currentdir + '/unique_' + str(i), ratios=False,
                    origin=False, colors=colors, dot_size=2, legend=False,
                    range_override=[-1, 3], transparent=True, connect_color='black')
