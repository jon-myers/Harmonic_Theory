import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import math, copy, json, pickle
import numpy as np
from utils import * 
# from plot import *
from build import build_sequence


def get_possible_octs(ratio, register):
    max_oct = np.floor(np.log2(register[1]/ratio))
    min_oct = np.ceil(np.log2(register[0]/ratio))
    out = ratio * (2 ** np.arange(min_oct, max_oct+1))
    return out

def melody_maker(ratios, size, register = (100, 400), morph_rule='nearest', 
    morph=None):
    """Returns a sequence of frequency values, stochastically chosen via dc_alg, 
    of notes from a given set of ratios, cast to within the desired register. 
    Always moves to either the nearest element above, or the nearest element 
    below, according to a morph or morph subroutine. If morph_rule is 'nearest', 
    always move to the nearest tone. 
    
    
    
    Parameters:
        ratios (array, floats, where 1<=x<2): octave generalized interval ratio
        size (integer): number of notes in the melody 
        register (tuple, with length = 2): min and max frequencies in register
        morph_rule (string): subroutine for morphology
            'nearest': sucessive notes are at nearest possible interval to 
                       previous notes.
            ''
    """
    indexes = dc_alg(len(ratios), size)
    sequence = ratios[indexes]
    start = np.random.choice(get_possible_octs(sequence[0], register))
    freq_sequence = [start]
    for rat in sequence[1:]:
        pos = get_possible_octs(rat, register)
        dist = np.abs(pos - freq_sequence[-1])
        freq = pos[np.argmin(dist)]
        freq_sequence.append(freq)
    return freq_sequence

chords = json.load(open('piece/branches/3dims.json', 'r'))

sequence = build_sequence(chords[1:-2], 20, 3, overlap_range = (0.8, 1.0))

primes = np.array((3, 11, 7), dtype=float)
rat_sequence = [hsv_to_gen_ratios(i, primes) for i in sequence]
ratios = rat_sequence[0]
melodies = [melody_maker(i, 1000) for i in rat_sequence]

json.dump(melodies, open('piece/sequences/melody_0.json', 'w'), cls=NpEncoder)
