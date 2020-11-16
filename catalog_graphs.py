import json
from matplotlib import pyplot as plt
import numpy as np
from utils import cast_to_ordinal, make_plot, are_roots, get_complement_shell 
from utils import make_shell_plot, draw_arc


primes = np.array((2, 3, 5))
with open('chords/chords4.json') as json_file:
    chords = json.load(json_file)
    all_roots = []
    for chord in chords:
        points = np.array(chord['points'])
        roots = cast_to_ordinal(points[are_roots(points)])
        if str(roots) not in [str(i) for i in all_roots]:
            all_roots.append(roots) 
    for i, roots in enumerate(all_roots):
        shell = get_complement_shell(roots)
        make_shell_plot(shell, roots, primes, 'chord_graphs/chord_4_' + str(i), legend=False, 
        ratios=False, origin=False, transparent=True, origin_range=[0, 4],
        point_color='indianred', shell_color='grey', range_override=[0, 3], 
        shell_dot_size=0.25, dot_size=2, angles=True)
