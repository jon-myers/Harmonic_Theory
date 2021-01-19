import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import json
import math
import numpy as np
from utils import *



def branch_sequence_step(branches, seed=None, ct_obj=None, overlap_range=(0.4, 0.8)):
    
    if ct_obj == None:
        size_ct = np.zeros(len(branches), dtype=int) + 1
        ind_cts = [np.zeros(len(i), dtype=int) + 1 for i in branches]
        perm_ct = np.zeros(math.factorial(np.shape(branches[0])[-1]), dtype=int)+1 
        # print(np.shape(branches[0])[-1], '\n\n')
        ct_obj = {'size': size_ct, 'seed_ind': ind_cts, 'perm': perm_ct}   
    
    size, ct_obj['size'] = dc_alg_step(len(branches), ct_obj['size'])
    ct_obj['seed_ind'] = [np.zeros(len(i), dtype=int)+1 for i in branches]
    seed_index, ct_obj['seed_ind'][size] = dc_alg_step(
                    len(branches[size]), ct_obj['seed_ind'][size])
    next = branches[size][seed_index]
    perms = get_transpositions(next)
    
    
    if np.all(seed == None):
        perm_index, ct_obj['perm'] = dc_alg_step(len(perms), ct_obj['perm'])
        seed = perms[perm_index]
        return seed, ct_obj
    else:
        # TODO pick up here ...
        overlap_min = np.int(np.ceil(overlap_range[0] * len(next)))
        overlap_max = np.int(np.ceil(overlap_range[1] * len(next)))
        print(overlap_min, overlap_max)
        
        
        
        # ct_obj = {'size': size_ct, 'seed_ind': ind_cts, 'perm': perm_ct}
        return seed, ct_obj
        
    # elif ct_obj == None:
    #     size_ct = np.zeros(len(branches), dtype=int) + 1
    #     ind_cts = [np.zeros(len(i), dtype=int) + 1 for i in branches]
    #     ct_obj = {'size': size_ct, 'seed_ind': ind_cts, 'perm': perm_ct}
        
    
    




chords_3d = json.load(open('piece/branches/3dims.json', 'r'))

seed, ct_obj = branch_sequence_step(chords_3d[1:])
next, ct_obj = branch_sequence_step(chords_3d[1:], seed, ct_obj)
