import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import math, copy, json, pickle
import numpy as np
from utils import *
# from plot import *

def branch_sequence_step(branches, seed=None, ct_obj=None, overlap_range=(0.2, 0.7)):
    if ct_obj == None:
        size_ct = np.zeros(len(branches), dtype=int) + 1
        ind_cts = [np.zeros(len(i), dtype=int) + 1 for i in branches]
        perm_ct = np.zeros(math.factorial(np.shape(branches[0])[-1]), dtype=int)+1
        ct_obj = {'size': size_ct, 'seed_ind': ind_cts}

    if np.all(seed == None):
        size, ct_obj['size'] = dc_alg_step(len(branches), ct_obj['size'])
        seed_index, ct_obj['seed_ind'][size] = dc_alg_step(
                        len(branches[size]), ct_obj['seed_ind'][size])
        next = branches[size][seed_index]
        perms = get_transpositions(next)
        seed = perms[np.random.choice(range(len(perms)))]
        return seed, ct_obj
    else:
        next = None
        ct__ = 0
        while np.all(next == None):
            ct__ += 1
            print(ct__)
            if ct__ > 1000:
                breakpoint()
            size, temp_size_ct = dc_alg_step(len(branches), ct_obj['size'])
            temp_seed_ind_ct = copy.copy(ct_obj['seed_ind'])
            seed_index, temp_seed_ind_ct[size] = dc_alg_step(
                            len(branches[size]), ct_obj['seed_ind'][size])
            next = branches[size][seed_index]
            perms = get_transpositions(next)

            overlap_min = np.int(np.ceil(overlap_range[0] * len(next)))
            overlap_max = np.int(np.ceil(overlap_range[1] * len(next)))
            if overlap_min == overlap_max:
                overlap = overlap_min
            else:
                overlap = np.random.choice(np.arange(overlap_min, overlap_max))

            next = try_to_fit(seed, next, overlap)
        ct_obj = {'size': temp_size_ct, 'seed_ind': temp_seed_ind_ct}
        return next, ct_obj

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

def group_adjacencies(possible_adjs, pa_cts, seed):
    """For a set of unique possible adjacencies, and their associated counts,
    split the possible_adjs into a list of arrays, each with a different number
    of associated counts, in descending order. (The input of this method is
    the output of `ext_adjacencies()`). Then, sort each group internally such
    that points that 'contain' seed points are ahead of points that do not
    'contain' seed points."""
    possible_adjs = possible_adjs[np.argsort(pa_cts)[::-1]]
    pa_cts = np.sort(pa_cts)[::-1]
    unq_pa_cts, each_cts = np.unique(pa_cts, return_counts=True)
    unq_pa_cts = unq_pa_cts[::-1]
    each_cts = each_cts[::-1]
    iter = 0
    grouped_pas = []
    for ec in each_cts:
        grouped_pas.append(possible_adjs[iter:iter+ec])
        iter += ec
    for g, group in enumerate(grouped_pas):
        bool_mask = np.zeros(len(group), dtype=int)
        for p, point in enumerate(group):
            bool_mask[p] = are_roots(np.concatenate(([point], seed)))[0]
        sorts = np.argsort(bool_mask)[::-1]
        grouped_pas[g] = group[sorts]
    return grouped_pas

def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def comb_partition(number, slots):
    answer = set()
    answer.add((number, ))
    for x in range(1, number):
        for y in comb_partition(number - x, slots):
            answer.add((x, ) + y)
    answer = [i for i in answer if len(i)<=slots]
    answer = sorted(answer, key=lambda x: len(x))
    grouped_answer = [[j for j in answer if len(j) == i] for i in range(len(answer[-1]))]
    return answer

def sorted_comb_partition(number, max_lens):
    slots = len(max_lens)
    answer = comb_partition(number, slots)
    grouped_answer = [[j for j in answer if len(j) == i+1] for i in range(len(answer[-1]))]
    for i in range(len(grouped_answer)):
        ga = [(*[i for i in a], *[0 for j in range(slots-len(a))]) for a in grouped_answer[i]]
        permutations = flatten([list(itertools.permutations(ga[k])) for k in range(len(ga))])
        grouped_answer[i] = np.unique(permutations, axis=0)
        for dim in range(slots)[::-1]:
            grouped_answer[i] = sorted(grouped_answer[i], key=lambda x: x[dim], reverse=True)
    answer = np.array(flatten(grouped_answer))
    for dim in range(slots)[::-1]:
        answer = np.array(sorted(answer, key=lambda x: x[dim], reverse=True))
    answer = answer[np.all(answer <= max_lens, axis=1)]
    return answer

def get_layer_trial_counts(groups, extra, limit=2):
    """For a set of grouped adjacencies, return a list of arrays filled with
    the number of adjacencies in each group to try when fitting the filler to
    the seed. List is in order of preference; if the first don't work, try the
    second, etc."""

    if extra > limit:
        extra = limit
    group_lens = [len(i) for i in groups]
    group_len_sums = [sum(group_lens[:i+1]) for i in range(len(groups))]

    out_group_lens = copy.copy(group_lens)
    for i, gls in enumerate(group_len_sums):
        if gls >= extra:
            if i == 0:
                out_group_lens = [extra, *[0 for i in range(len(groups)-1)]]
                break
            else:
                add = extra - group_len_sums[i-1]
                out_group_lens = out_group_lens[:i] + [add] + [0 for i in range(len(groups) - (i+1))]
                break
    out_group_lens = sorted_comb_partition(sum(out_group_lens), group_lens)
    return out_group_lens

def specific_fit(perms, trial_points, chord_points, overlap):
    pos = []
    for p, perm in enumerate(perms):
        # print(trial_points)
        translations = multi_overlay_translations(trial_points, perm)
        for t, trans in enumerate(translations):
            intersect = npi.intersection(trans, chord_points)
            if len(intersect) == overlap:
                pos.append(reorder_points(trans))
    pos = npi.unique(pos)
    if len(pos) > 1:
        choice_index = np.random.choice(np.arange(len(pos)))
        choice = pos[choice_index]

        return choice
    elif len(pos) == 1:
        choice = pos[0]
        return choice
    else:
        return None

def try_to_fit(points, filler, overlap):
    perms = get_transpositions(filler)
    possible_adjs, pa_cts = ext_adjacencies(points)
    groups = group_adjacencies(possible_adjs, pa_cts, points)
    extra = len(filler) - overlap
    partitions = get_layer_trial_counts(groups, extra)
    for par_index, par in enumerate(partitions):
        trial_ex_points = []
        for l, layer_ct in enumerate(par):
            if layer_ct == 1:
                choice_index = np.random.choice(range(len(groups[l])))
                choice = np.array([groups[l][choice_index]])
                if len(np.shape(choice)) < 3:
                    choice = np.expand_dims(choice, 0)
                trial_ex_points.append(np.array(choice))
            elif layer_ct > 1:
                combs = list(itertools.combinations(range(len(groups[l])), layer_ct))
                ar_bool_mask = np.zeros((len(combs), layer_ct), dtype=bool)
                for i, comb in enumerate(combs):
                    for j, pt_idx in enumerate(comb):
                        points_plus = np.concatenate(([groups[l][pt_idx]], points))
                        ar_bool_mask[i, j] = are_roots(points_plus)[0]
                sorts = np.argsort(np.count_nonzero(ar_bool_mask, axis=1))[::-1]
                combs = np.array(combs)[sorts]
                trial = groups[l][combs]
                trial_ex_points.append(trial)
        if len(trial_ex_points) == 1:
            if len(np.shape(trial_ex_points[0])) == 1:
                choice = specific_fit(perms, trial_ex_points, points, overlap)
                if np.all(choice != None):
                    return choice
            else:
                for trial_points in trial_ex_points[0]:
                    choice = specific_fit(perms, trial_points, points, overlap)
                    if np.all(choice != None):
                        return choice
        elif len(trial_ex_points) > 1:
            cp_indexes = cartesian_product(*[np.arange(len(i)) for i in trial_ex_points])
            for i, index_list in enumerate(cp_indexes):
                if i > 600:
                    breakpoint()
                trial_points = []
                for j, index in enumerate(index_list):
                    trial_points.append(trial_ex_points[j][index])
                trial_points = np.concatenate(trial_points)
                choice = specific_fit(perms, trial_points[:3], points, overlap)
                if np.all(choice != None):
                    return choice
    return None

def overlay_translations(point, chord):
    """Given a fixed point and a chord in some shared set of dimensions,
    return all of the possible chord translations that intersect the point."""
    vecs = point - chord
    # print(vecs)
    sh = np.shape(chord)
    exp_chord = np.broadcast_to(chord, (sh[0], sh[0], sh[1]))
    exp_vecs = np.expand_dims(vecs, 1)
    exp_vecs = np.broadcast_to(exp_vecs, (sh[0], sh[0], sh[1]))
    return exp_chord + exp_vecs

def multi_overlay_translations(points, chord):
    """Given a fixed set of points and a chord in some shared set of dimensions,
    return all of the possible chord translations that intersect the points."""
    vecs = [point - chord for point in points]
    shared_vecs = npi.intersection(*vecs)
    sv_shape = np.shape(shared_vecs)
    ch_shape = np.shape(chord)
    sh = (sv_shape[0], ch_shape[0], ch_shape[-1])
    exp_chord = np.broadcast_to(chord, sh)
    exp_sv = np.expand_dims(shared_vecs, 1)
    exp_sv = np.broadcast_to(exp_sv, sh)
    return exp_chord + exp_sv

def build_sequence(chords, epochs, max_overlay, overlap_range):
    """


    if overlap_range is a list, rather than a tuple, its len must be one less
    than the number of epochs. """
    seed, ct_obj = branch_sequence_step(chords)
    overlay = [seed]
    succession = [seed]
    for i in range(epochs-1):
        if type(overlap_range) == list:
            overlap_ = overlap_range[i]
        else:
            overlap_ = overlap_range
        seed = np.concatenate(overlay)
        next, ct_obj = branch_sequence_step(chords, seed, ct_obj)
        overlay.append(next)
        succession.append(next)
        if len(overlay) > max_overlay:
            del overlay[0]
    return succession



# chords = json.load(open('piece/branches/3dims.json', 'r'))
#
# succession = build_sequence(chords[1:-2], 20, 4, overlap_range = (0.8, 1.0))
#
# pickle.dump(succession, open('piece/sequences/sequence_0.p', 'wb'))
# for i, o in enumerate(overlay):
#     make_plot(o, 'piece/plots/' + str(i), origin=True, connect_color='black')
#     seed = np.concatenate(overlay[:i+1])
#     make_plot(seed, 'piece/plots/iterative_' + str(i), origin=True, connect_color='black')
#     print(o)
#     print()
