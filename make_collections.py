import numpy as np
import numpy_indexed as npi
from scipy.spatial.distance import cdist
from utils import cast_to_ordinal, plot_basic_hsl, reorder_points, \
    cartesian_product, sub_branches, unique_sub_branches, containment_size, \
    get_stability, get_loops, NpEncoder, get_routes, get_transposition_shell, \
    get_multipath_shell, mean_root_distance, mean_root_angle, are_roots
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
    for tones in range(max_tones-1):
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
    return chords[1:]

def pack_stats(chords, path):
    objs = []
    for c, chord in enumerate(chords):
        t_shell = get_transposition_shell(chord)
        m_shell = get_multipath_shell(chord)
        roots = chord[are_roots(chord)]

        obj = {}
        obj['points'] = chord
        obj['index'] = c
        obj['branch_decomposition_size'] = len(sub_branches(chord))
        obj['unique_branch_decomposition_size'] = len(unique_sub_branches(chord))
        obj['containment_size'], obj['containment_index'] = containment_size(chord)
        obj['stability'] = get_stability(chord)
        obj['loops'] = get_loops(chord)
        obj['routes'] = get_routes(chord)
        obj['transposition_shell_size'] = len(t_shell)
        obj['transposition_shell_proportion'] = len(chord) / len(t_shell)
        obj['multipath_shell_size'] = len(m_shell)
        obj['multipath_shell_proportion'] = len(chord) / len(m_shell)
        obj['roots'] = roots
        obj['mean_root_distance'] = mean_root_distance(chord)
        obj['mean_root_angle'] = mean_root_angle(chord)

        objs.append(obj)
    json.dump(objs, open(path, 'w'), cls=NpEncoder)

chords = make_chords(6, 4)

# print(chords[:2])
pack_stats(chords[-1], 'etudes/chords.json')
#
# json.dump(chords, open(''))
# test_chords = json.dump(chords, open('test.json', 'w'), cls=NpEncoder)
# # for chord in chords:
# #     print(chord)
# # c = chords[-1]
# # print(len(c))

# pack_stats(c, 'test.json')
