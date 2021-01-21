import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import math, copy, json, pickle
import numpy as np
from utils import *



def branch_sequence_step(branches, seed=None, ct_obj=None, overlap_range=(0.4, 0.8)):

    if ct_obj == None:
        size_ct = np.zeros(len(branches), dtype=int) + 1
        ind_cts = [np.zeros(len(i), dtype=int) + 1 for i in branches]
        perm_ct = np.zeros(math.factorial(np.shape(branches[0])[-1]), dtype=int)+1
        # print(np.shape(branches[0])[-1], '\n\n')
        ct_obj = {'size': size_ct, 'seed_ind': ind_cts}

    # size, ct_obj['size'] = dc_alg_step(len(branches), ct_obj['size'])
    # temp_seed_ind_ct = copy.copy(ct_obj['seed_ind'])
    # seed_index, temp_seed_ind_ct[size] = dc_alg_step(
    #                 len(branches[size]), ct_obj['seed_ind'][size])
    # next = branches[size][seed_index]
    # perms = get_transpositions(next)

    if np.all(seed == None):
        size, ct_obj['size'] = dc_alg_step(len(branches), ct_obj['size'])
        seed_index, ct_obj['seed_ind'][size] = dc_alg_step(
                        len(branches[size]), ct_obj['seed_ind'][size])
        next = branches[size][seed_index]
        perms = get_transpositions(next)
        seed = perms[np.random.choice(range(len(perms)))]
        return seed, ct_obj
    else:
        # TODO pick up here ...

        next = None
        while np.all(next == None):
            size, temp_size_ct = dc_alg_step(len(branches), ct_obj['size'])
            temp_seed_ind_ct = copy.copy(ct_obj['seed_ind'])
            seed_index, temp_seed_ind_ct[size] = dc_alg_step(
                            len(branches[size]), ct_obj['seed_ind'][size])
            next = branches[size][seed_index]
            perms = get_transpositions(next)

            overlap_min = np.int(np.ceil(overlap_range[0] * len(next)))
            overlap_max = np.int(np.ceil(overlap_range[1] * len(next)))
            overlap = np.random.choice(np.arange(overlap_min, overlap_max))

            next = try_to_fit(seed, next, overlap)
        ct_obj = {'size': temp_size_ct, 'seed_ind': temp_seed_ind_ct}
        return seed, ct_obj

    # elif ct_obj == None:
    #     size_ct = np.zeros(len(branches), dtype=int) + 1
    #     ind_cts = [np.zeros(len(i), dtype=int) + 1 for i in branches]
    #     ct_obj = {'size': size_ct, 'seed_ind': ind_cts, 'perm': perm_ct}






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
    
# scp = sorted_comb_partition(7, [1, 3, 5, 7])
# print(scp)
def get_layer_trial_counts(groups, extra):
    """For a set of grouped adjacencies, return a list of arrays filled with
    the number of adjacencies in each group to try when fitting the filler to
    the seed. List is in order of preference; if the first don't work, try the
    second, etc."""

    group_lens = [len(i) for i in groups]
    group_len_sums = [sum(group_lens[:i+1]) for i in range(len(groups))]

    out_group_lens = copy.copy(group_lens)
    # print('gls', group_lens, group_len_sums)
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

test = [np.array((0, 1, 2)), np.array((-1, 0, 1))]





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
    # print(points, '\n', filler, '\n', overlap)
    perms = get_transpositions(filler)
    possible_adjs, pa_cts = ext_adjacencies(points)
    groups = group_adjacencies(possible_adjs, pa_cts, points)
    extra = len(filler) - overlap
    partitions = get_layer_trial_counts(groups, extra)
    for par in partitions:
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
                # print(ar_bool_mask, sorts, '\n\n')
        if len(trial_ex_points) == 1:
            # print(trial_ex_points)
            # print(np.shape(trial_ex_points[0]))
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
                trial_points = []
                for j, index in enumerate(index_list):
                    trial_points.append(trial_ex_points[j][index])
                trial_points = np.concatenate(trial_points)
                choice = specific_fit(perms, trial_points, points, overlap)
                if np.all(choice != None):
                    return choice    
    return None
            
            
    #         # print(trial_ex_points)
    #     # print('trial points: \n', len(trial_ex_points), '\n\n')
    #     # for layer in trial_ex_points:
    #     #     for trial in layer:
    # 
    # 
    # 
    # print('\n\n')
    # 
    # 
    # #TODO pick things up from here. For each in gl
    # 
    # 
    # 
    # #
    # # group_lens = [len(i) for i in groups]
    # # group_len_sums = [sum(group_lens[:i]) for i in range(1, len(groups))]
    # #
    # # out_group_lens = copy.copy(group_lens)
    # # for i, gls in enumerate(group_len_sums):
    # #     if gls > extra:
    # #         if i == 0:
    # #             out_group_lens = [[extra], *[0 for i in range(len(groups)-1)]]
    # #             return out_group_lens
    # #         else:
    # #             add = extra - gls[i-1]
    # #             out_group_lens[i] = add
    # #             out_group_lens = out_group_lens[:i+1] + [0 for i in range(len(group) - (i+1))]
    # #
    # # iter = 0
    # # while iter > extra:
    # 
    # 
    # 
    # 
    # 
    # 
    # possible_adjs = possible_adjs[np.argsort(pa_cts)[::-1]]
    # pa_cts = np.sort(pa_cts)[::-1]
    # act_pas = possible_adjs[pa_cts > 1] #actionable Possible adjacencies
    # extra = len(filler) - overlap
    # if len(act_pas) >= extra:
    #     if extra >= 1:
    #         trans_list = []
    #         combs = list(itertools.combinations(range(len(act_pas)), extra))
    #         comb_bool_mask = np.zeros(len(combs), dtype=bool)
    #         for c, comb in enumerate(combs):
    #             ext_pts = np.array([act_pas[i] for i in comb])
    #             pos = []
    #             for p, perm in enumerate(perms):
    #                 translations = multi_overlay_translations(ext_pts, perm)
    #                 for t, trans in enumerate(translations):
    #                     intersect = npi.intersection(trans, points)
    #                     if len(intersect) == overlap:
    #                         pos.append(reorder_points(trans))
    #             pos = npi.unique(pos)
    #             if len(pos) >= 1:
    #                 comb_bool_mask[c] = True
    #                 for po in pos:
    #                     trans_list.append(po)
    # 
    #         if len(trans_list) == 0:
    #             return None
    #         elif len(trans_list) == 1:
    #             return trans_list[0]
    #         elif len(trans_list) > 1:
    #             return trans_list[np.random.choice(np.arange(len(trans_list)))]
    # 
    #     # elif extra == 1:
    #     #     ext_pt = possible_adjs[0]
    #     #     bool_mask = np.zeros((len(perms), len(points)), dtype=bool)
    #     #     for p, perm in enumerate(perms):
    #     #         translations = overlay_translations(ext_pt, perm)
    #     #         for t, trans in enumerate(translations):
    #     #             intersect = npi.intersection(trans, points)
    #     #             if len(intersect) == overlap:
    #     #                 bool_mask[p, t] = True
    #     #
    #     #     possible_perms = np.unique(np.nonzero(bool_mask)[0])
    #     #     if len(possible_perms) == 0 :
    #     #         return None
    #     #     else:
    #     #         perm_index = np.random.choice(possible_perms)
    #     #         seive = np.nonzero(bool_mask)
    #     #         t_inds = [t for i, t in enumerate(seive[1]) if seive[0][i] == perm_index]
    #     #         trans_index = np.random.choice(t_inds)
    #     #         trans = overlay_translations(ext_pt, perms[perm_index])[trans_index]
    #     #         print(ext_pt)
    #     #         print(trans)
    # 
    #         # print(bool_mask)
    #         # print(np.nonzero(bool_mask))
    #         # print(bool_mask[np.nonzero(bool_mask)])
    # 
    #     # combs = itertools.combinations(range(len(actionble_pas)), extra)

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
    # print(points)
    # print(chord)
    # print()
    vecs = [point - chord for point in points]
    shared_vecs = npi.intersection(*vecs)
    # if len(shared_vecs) == 0:
    #     return None
    # else:
    sv_shape = np.shape(shared_vecs)
    ch_shape = np.shape(chord)
    sh = (sv_shape[0], ch_shape[0], ch_shape[-1])
    exp_chord = np.broadcast_to(chord, sh)
    exp_sv = np.expand_dims(shared_vecs, 1)
    exp_sv = np.broadcast_to(exp_sv, sh)
    return exp_chord + exp_sv


    # sh = np.shape(chord)
    # exp_chord = np.broadcast_to(chord, (sh[0], sh[0], sh[1]))
    # exp_vecs = np.expand_dims(vecs, 1)
    # exp_vecs = np.broadcast_to(exp_vecs, (sh[0], sh[0], sh[1]))
    # return exp_chord + exp_vecs



#


# test = np.array(chords_3d[-3][1])
# filler = np.array(chords_3d[-6][0])
#
# # print(test, '\n\n', filler, '\n\n')
# # print()
#
#
#
# pa, cts = ext_adjacencies(test)
# two_pts = pa[cts==2][1:]
# test_pts = np.array(((0, 1, 1), (0, 2, 1)))
# print(test, '\n\n', test_pts, '\n\n')
# mot = multi_overlay_translations(test_pts, test)
# print(mot)
# print(cts[np.argsort(cts)[::-1]])
# out = try_to_fit(test, filler, len(filler) - 1)
# print(out)




# 
# 

chords_3d = json.load(open('piece/branches/3dims.json', 'r'))
seed, ct_obj = branch_sequence_step(chords_3d[1:])
# pickle.dump((seed, ct_obj), open('piece/troubleshoot.p', 'wb'))
# 
# pick = pickle.load(open('piece/troubleshoot.p', 'r'))
# seed = pick[0]
# ct_obj = pick[1]
next, ct_obj = branch_sequence_step(chords_3d[1:], seed, ct_obj)
breakpoint()
print(next)
