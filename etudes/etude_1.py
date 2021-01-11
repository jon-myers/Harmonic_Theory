#baby steps ...

# Etude 1: X number of chords, each of which shares a root or root adjacency
# with at least one other chord.
import sys
from os import path
sys.path.append(path.dirname((path.dirname(path.abspath(__file__)))))
import json
import numpy as np
from utils import *

# print(t)

def make_random_trajectory(length, max_step=1, dims=3, circular=True):
    """

    parameters:
        length (integer, if circular==True, must be even)

    """
    steps = np.zeros((length, dims), dtype=int)
    indexes = np.random.randint(dims, size=length)
    steps[np.arange(length), indexes] = np.random.randint(-1, 2, size=length)
    if circular == True:
        mirror = -1 * steps[:int(length/2)]
        np.random.shuffle(mirror)
        steps[int(length/2):] = mirror
    return steps


def traj_to_absolute(traj):
    origin = np.zeros(shape=(1, np.shape(traj)[-1]), dtype=int)
    return np.concatenate(((origin), np.cumsum(traj, axis=0)))


def hsv_to_freq(hsv, primes, fund, oct=(0, 0, 0)):
    oct = np.array(oct)

    if len(np.shape(hsv)) == 2:
        sub_prod = (primes ** hsv) * (2.0 ** (hsv * oct))
        freq = fund * np.prod(sub_prod, axis=1)
    else:
        freq = fund * np.prod((primes ** hsv) * (2.0 ** (hsv * oct)))
    return freq

def octave_finder(chord, fund, primes, lims = (50, 50 * (2 ** 5))):
    """Returns all of the possible octave shifts that would make the notes in
    the chord audible."""
    bit = np.arange(-4, 4)
    cp = cartesian_product(*(bit for i in range(np.shape(chord)[-1])))
    seive = np.zeros(len(cp), dtype=bool)
    for i, oct in enumerate(cp):
        freq = hsv_to_freq(chord, primes, fund, oct)
        if np.all(np.min(freq) >= lims[0]) and np.all(np.max(freq) <= lims[1]):
            seive[i] = True
    possible_octs = cp[seive]

    return possible_octs

def make_chord_sequence(size, fund=200, primes=(3.0, 5.0, 7.0), min_comma=40,
                        lims=(100, 100*(2**3)), chord_size=5):
    primes = np.array(primes)
    t = make_random_trajectory(size-1)
    a = traj_to_absolute(t)
    c = json.load(open('etudes/chords.json', 'r'))
    chords = np.array([i['points'] for i in c])
    roots = [i['roots'] for i in c]
    indexes = np.random.randint(len(chords), size=size)
    chords = chords[indexes]
    oct_shifts = []
    for i, chord in enumerate(chords):
        trs = get_transpositions(chord)
        tr = trs[np.random.choice(len(trs))]
        chords[i] = tr
        roots = tr[are_roots(tr)]
        root_index = np.random.choice(np.arange(len(roots)))
        root = roots[root_index]
        diff = a[i] - root
        chords[i] += diff
        octs = octave_finder(chords[i], fund, primes, lims)

        # minimize commas by filtering list of possible octaves even
        # further to stop including those whose minimum of combinatorial
        # magnitude differences of frequencies is above some threshold.
        min_cents = []
        for oct in octs:
            freqs = hsv_to_freq(chords[i], primes, fund, oct)
            combs = itertools.combinations(range(len(freqs)), 2)
            diffs = [hz_to_cents(freqs[c_[0]], freqs[c_[1]]) for c_ in list(combs)]
            min = np.min(np.abs(diffs))
            min_cents.append(min)
        octs = octs[np.array(min_cents) >= min_comma]

        # choose between the different octs, randomly
        oct = octs[np.random.randint(len(octs))]
        oct_shifts.append(oct)
    return chords, np.array(oct_shifts)



fund = 200
primes = np.array((3.0, 5.0, 7.0))
chords, oct_shifts = make_chord_sequence(13, fund)
all_freq = []

for i, chord in enumerate(chords):
    freqs = hsv_to_freq(chord, primes, fund, oct_shifts[i])
    all_freq.append(freqs)

all_freq = np.round(all_freq, 2)
json.dump(all_freq, open('etudes/chord_sequence.json', 'w'), cls=NpEncoder)

# sb = sub_branches(chords[0])
for chord in chords:
    usb, matches = unique_sub_branches(chord, count=True)
    lens = [len(i) for i in matches]
    # print(lens)
    max_lens = max([len(i) for i in matches[:-1]])
    print(max_lens)
    if max_lens > 4:
        print(chord)

    
    
    # print()
# usb = flatten(usb)
# idx = [cast_to_ordinal(i) for i in sb]
# print(idx, '\n\n', cts, '\n\n')
# for i in range(len(usb)):
#     print(usb[i])
#     print(idx[i])
#     print(cts[i])
#     print()
# for i in range(len(usb)):
# #     print(usb[i])
# for i in range(len(sb)):
#     print(sb[i], '\n\n')
#     print(usb[i], '\n\n\n')
