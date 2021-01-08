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

def octave_finder(chord, fund, primes, lims = (60, 50 * 2 ** 5)):

    bit = np.arange(-4, 4)
    cp = cartesian_product(*(bit for i in range(np.shape(chord)[-1])))
    seive = np.zeros(len(cp), dtype=bool)
    for i, oct in enumerate(cp):
        freq = hsv_to_freq(chord, primes, fund, oct)
        if np.all(np.min(freq) >= lims[0]) and np.all(np.max(freq) <= lims[1]):
            seive[i] = True
    possible_octs = cp[seive]
    return possible_octs

    """Returns all of the possible octave shifts that would make the notes in
    the chord audible."""

size = 13
t = make_random_trajectory(size-1)
a = traj_to_absolute(t)
primes = np.array((3.0, 5.0, 7.0))
fund = 200
# octs = TODO make octfinder !!
c = json.load(open('etudes/chords.json', 'r'))
chords = np.array([i['points'] for i in c])
roots = [i['roots'] for i in c]

indexes = np.random.randint(len(chords), size=size)
chords = chords[indexes]
for i, chord in enumerate(chords):
    trs = get_transpositions(chord)
    tr = trs[np.random.choice(len(trs))]
    chords[i] = tr
    roots = tr[are_roots(tr)]
    root = roots[np.random.choice(np.arange(len(roots)))]
    diff = a[i] - root
    chords[i] += diff


octs = octave_finder(chords[0], 200, primes)
print(len(octs))
# freqs = hsv_to_freq(chords[i], primes, fund)


#
# print(chords)
# print(freqs)
