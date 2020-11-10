import json
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, get_transpositions, are_roots, cast_to_ordinal, get_stability

primes = np.array((2, 3, 5))
with open('chords/chords5.json') as json_file:
    branches = json.load(json_file)
    branch_array = []
    for i in range(1, 7):
        sub_array = []
        for branch in branches:
            if branch['paths'] == i:
                sub_array.append(branch['points'])
        branch_array.append(np.array(sub_array[np.random.choice(np.arange(len(sub_array)))]))
    for i, branch in enumerate(branch_array):
        choices = ['black', 'red']
        colors = [choices[int(k)] for k in are_roots(branch)]
        make_plot(branch, primes, currentdir + '/' + str(i+1), dot_size=1,
                  legend=False, ratios=False, origin=False, transparent=True,
                  connect_color='black', range_override=[0, 4], colors=colors,
                  connect_size=3, opacity=1)
