import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import json
import numpy as np
from utils import *



def branch_sequence_step(branches, start=None, size_range=(2, 5),
                        overlap_range=(0.8, 1.0)):
    if np.all(start==None):
        i = np.random.choice(np.arange(len(chords)))
        j = np.random.choice(np.arange(len(chords[i])))
        start = branches[i][j]

    for s in range(*size_range):
        overlap_min = round(s * overlap_range[0])
        overlap_max = round(s * overlap_range[1])



# already did the first half, now need to take a shape, get all non-member
# adjacencies, and see how many of the norms each of those adjacencies is
# adjacent to

def ext_adjacencies(points):
    """For a set of points, find all external unique adjacencies to those points
    that are not included in the set of points. And, return an
    array of counts of how many points in the original set of points each
    ext_adjacency is adjacent to."""
    points = np.array(points)
    dirs = np.eye(np.shape(points)[-1], dtype=int)
    dirs = np.concatenate((dirs, -1*dirs))
    possible_adjs = np.expand_dims(points, 1)
    sh = np.shape(possible_adjs)
    possible_adjs = np.broadcast_to(possible_adjs, (sh[0], len(dirs), sh[2]))
    possible_adjs = (possible_adjs + dirs).reshape((sh[0] * len(dirs), sh[2]))
    possible_adjs = possible_adjs[np.invert(npi.contains(points, possible_adjs))]
    return npi.unique(possible_adjs, return_count=True)

# next piece is to fill any points that are adjacencies.

def try_to_fit(points, filler, overlap):
    perms = get_transpositions(filler)
    possible_adjs, pa_cts = ext_adjacencies(points)
    possible_adjs = possible_adjs[np.argsort(pa_cts)[::-1]]
    pa_cts = np.sort(pa_cts)[::-1]
    actionable_pas = np.count_nonzero(pa_cts > 1)
    if actionable_pas >= len(points) - overlap:



#
chords_3d = json.load(open('piece/branches/3dims.json', 'r'))

test = chords_3d[-3][1]
filler = chords_3d[-5][0]
print(filler)

pa, cts = ext_adjacencies(test)
# print(pa), print(cts)
# print(cts[np.argsort(cts)[::-1]])
try_to_fit(test, filler, 1)
