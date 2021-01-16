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




chords_3d = json.load(open('piece/branches/3dims.json', 'r'))


chord_sequence(chords_3d)
