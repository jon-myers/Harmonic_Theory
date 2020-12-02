import numpy as np
import numpy_indexed as npi
from utils import cast_to_ordinal, plot_basic_hsl, reorder_points, \
    cartesian_product, sub_branches, unique_sub_branches, containment_size
import json, itertools

def make_branches(max_tones, dims):
    branches = [np.array([[np.repeat(0, dims)]])]
    for tones in range(max_tones):
        seeds = branches[-1]
        sub_branches = None
        for seed in seeds:
            add = np.eye(dims, dtype=int)
            shape = np.shape(seed)
            out = np.expand_dims(seed, 1)
            out = np.broadcast_to(out, (shape[0], dims, shape[1]))
            out = np.concatenate(out+add)
            filter = npi.contains(seed, out)
            out = out[np.invert(filter)]
            next = np.array([reorder_points(cast_to_ordinal(np.concatenate((seed, [i])))) for i in out])
            next = npi.unique(next)
            if 'sub_branches' in locals() and np.all(sub_branches != None):
                sub_branches = npi.unique(np.concatenate((sub_branches, next)))
            else:
                sub_branches = next
        sub_branches = np.array([reorder_points(i) for i in sub_branches])
        sub_branches = npi.unique(sub_branches)
        branches.append(sub_branches)
    return branches

def make_chords(max_tones, dims):
    chords = [np.array([[np.repeat(0, dims)]])]
    for tones in range(max_tones):
        seeds = chords[-1]
        sub_chords = None
        for seed in seeds:
            add = np.concatenate((np.eye(dims, dtype=int), -1 * np.eye(dims, dtype=int)))
            shape = np.shape(seed)
            out = np.expand_dims(seed, 1)
            out = np.broadcast_to(out, (shape[0], dims*2 , shape[1]))
            out = np.concatenate(out+add)
            filter = npi.contains(seed, out)
            out = out[np.invert(filter)]
            next = np.array([reorder_points(cast_to_ordinal(np.concatenate((seed, [i])))) for i in out])
            next = npi.unique(next)
            if 'sub_chords' in locals() and np.all(sub_chords != None):
                sub_chords = npi.unique(np.concatenate((sub_chords, next)))
            else:
                sub_chords = next
        sub_chords = np.array([reorder_points(i) for i in sub_chords])
        sub_chords = npi.unique(sub_chords)
        chords.append(sub_chords)
    return chords

chords = make_chords(5, 3)

c = chords[-1][-1]
branches = sub_branches(c)
unq_branches = unique_sub_branches(c)
print(len(branches))
print(len(unq_branches))

# print(len(branches))
# hash = [str(branches)]
# print(len(npi.unique(branches)))

def pack_stats(chords, path):
    objs = []
    for c, chord in enumerate(chords):
        obj = {}
        obj['points'] = chord
        obj['index'] = c
        obj['branch_decomposition_size'] = len(sub_branches(c))
        obj['unique_branch_decomposition_size'] = len(unique_sub_branches(c))
        obj['containment_size'], obj['containment_index'] = containment_size(c)
        objs.append(obj)


# the other way of doing things seems to leave some out. I feel more confident
# about this way of doing things; certainly seems faster so far. Next thing should
# be pretty similar
