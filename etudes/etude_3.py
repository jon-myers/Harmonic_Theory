import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import json
import numpy as np
from utils import *
from numpy_indexed.arraysetops import _set_count as set_count
# import numpy_indexed as npi


c = json.load(open('etudes/chords.json', 'r'))
containments = [i['containment_size'] for i in c]
stab = [i['stability'] for i in c]
print(np.max(stab))
print(np.min(stab))
max_containments = np.max(containments)
print(max_containments)

chords = []
for i, item in enumerate(c):
    if stab[i] < 2 * np.min(stab):
        chords.append(item['points'])
A = chords[np.random.choice(range(len(chords)))]
perms = get_transpositions(A)
primes = np.array((3, 5, 7, 11))
fund = 1
octs = [octave_finder(perm, fund, primes, lims=(1, 2**5)) for perm in perms]
octs = [octs_[np.random.choice(range(len(octs_)))] for octs_ in octs]
print(octs)
# sc = [set_count(octs, i) for i in range(1, 7)]
# sc = [i for i in sc if len(i) > 0]
# oct = sc[-1][np.random.choice(range(len(sc[-1])))]
# print(npi.intersection(octs[1], octs[2]))
freqs = [hsv_to_freq(perms[i], primes, fund, octs[i]) for i in range(len(octs))]
for freq in freqs:
    print(freq)
# print()
freqs = [np.array(fill_in_octaves(freq, 300)) for freq in freqs]
# for freq in freqs:
#     print(np.round(freq, 3))
# print([np.round(freq, 2) for freq in freqs])



      
json.dump(freqs, open('etudes/etude_3_seq.json', 'w'), cls=NpEncoder)
