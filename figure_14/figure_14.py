import json
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import make_plot, get_transpositions, are_roots, cast_to_ordinal, get_stability

primes = np.array((2, 3, 5))
with open('branches/branches3.json') as json_file:
    branches = json.load(json_file)
    branch_array = []
    for b in branches:
        branch_array.append(cast_to_ordinal(np.array(b['points'])))
    branch_array = np.array(branch_array)
    vals = np.array([get_stability(branch) for branch in branch_array])
    sorts = np.argsort(vals)[::-1]
    for i, branch in enumerate(branch_array[sorts]):
        choices = ['black', 'red']
        colors = [choices[int(i)] for i in are_roots(branch)]
        print(vals[sorts[i]])
        make_plot(branch, primes, currentdir + '/' + str(i), dot_size=5,
                  legend=False, ratios=False, origin=False, transparent=True,
                  connect_color='black', colors=colors, range_override=[0, 3])
