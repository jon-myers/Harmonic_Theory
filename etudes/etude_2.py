# with at least one other chord.
import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import json
import numpy as np
from utils import *

c = json.load(open('etudes/chords.json', 'r'))
chords = np.array([i['points'] for i in c])
br_indexes = np.array([np.count_nonzero(are_roots(chord)) == 1 for chord in chords])
chords = chords[br_indexes]
chord = chords[np.random.choice(np.arange(len(chords)))]

sb = sub_branches(chord)

sequence = group_dc_alg(sb, 100)
fund = 250
primes = np.array((3, 5, 7))
octaves, freq_ranges = octave_finder(chord, fund, primes, max_width=True)
octave = octaves[np.random.choice(range(len(octaves)))]

sequence = [np.round(hsv_to_freq(i, primes, fund, octave), 2) for i in sequence]
json.dump(sequence, open('etudes/etude_2_seq.json', 'w'), cls=NpEncoder)
