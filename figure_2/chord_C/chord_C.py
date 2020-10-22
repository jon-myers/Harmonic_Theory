import os,sys,inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from utils import make_plot, hz_to_cents, cartesian_product
import abjad, more_itertools
import numpy_indexed as npi

print('this:', hz_to_cents(11/8, 1))
# notes = ["f,4", "c4", "e4", "g4", "d'4"]
# notes = ["c'4", "g'4", "b'4", "d''4", "a''4"]
notes = ["c'4", "f'4", "fs'4", "cs''4"]
notes = [abjad.Note(i) for i in notes]

time_signature = abjad.TimeSignature((5, 4))

devs = [-2, -49, -47]
devs = [abjad.Markup(str(i)).tiny() for i in devs]
spacer = abjad.Markup.hspace(0.9)
markup_list = more_itertools.intersperse(spacer, devs)
abjad.attach(abjad.Markup.concat(markup_list, direction=abjad.Down).raise_(1.5), notes[1])

staff = abjad.Staff(notes)
staff.remove_commands.append('Time_signature_engraver')
abjad.attach(time_signature, staff[0])

path = currentdir + '/chord_C'
abjad.persist(staff).as_pdf(pdf_file_path=path)
abjad.persist(staff).as_ly(path+'.ly')

points = np.array((
[0, 0, 0],
[2, -1, 0],
[-3, 0, 1],
[-4, 1, 1]
))

def get_dense_extras(points):
    maxs = np.max(points, axis=0) + 1
    mins = np.min(points, axis=0)
    ranges = [np.arange(mins[i], maxs[i]) for i in range(3)]
    prod = cartesian_product(*ranges)
    inds = npi.contains(points, prod)
    return prod[np.invert(inds)]
    
draw_points = get_dense_extras(points)
draw_points = np.array((
[1, 0, 0],
[2, 0, 0],
[0, -1, 0],
[1, -1, 0],
[2, -1, 0],
[-1, 0, 0],
[-2, 0, 0],
[-3, 0, 0],
[-4, 0, 0],

[0, 1, 0],
[-1, 1, 0],
[-2, 1, 0],
[-3, 1, 0],
[-4, 1, 0],
[-4, 0, 1],
[-3, 1, 1]

))

# draw_points = np.array((
# [1, 0, 0],
# [2, 0, 0],
# [0, -1, 0],
# [1, -1, 0],
# [3, -1, 0],
# [4, -1, 0],
# [2, -2, 0], 
# [3, -2, 0],
# [-1, 0, 0],
# [-2, 0, 0],
# [-3, 0, 0], 
# [0, 0, 1],
# [-1, 0, 1],
# [-2, 0, 1],
# [-3, 1, 1],
# [-1, 1, 1],
# [-1, -1, 0],
# [-2, 1, 0],
# [-1, 1, 0],
# [-3, 1, 0],
# [0, 1, 0],
# [0, 1, 1]
# ))
primes = np.array([2.0, 3.0, 11])


# points = np.roll(points, 2, axis=1)
# draw_points = np.roll(draw_points, 2, axis=1)
# primes = np.roll(primes, 2)

make_plot(points, primes, 'figure_2/chord_C/chord_C_plot', draw_points = draw_points)
