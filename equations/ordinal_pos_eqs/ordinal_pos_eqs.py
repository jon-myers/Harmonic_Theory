import os, sys, inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import tex_matrix_writer, cast_to_ordinal
from plot import make_plot, make_2d_plot
A = np.array(((0, 0), (0, 1), (0, 2), (1, 2)))
A_sorted = cast_to_ordinal(A)

one = np.array((
(-1, 0, 1),
(-1, 0, 0),
(-1, -1, 0),
(-1, -1, -1),
(0, -1, 0),
(1, -1, 0),
(1, -1, 1)
))
one_sorted = one - np.min(one, axis=0)

two = np.array((
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2)))

two_sorted = cast_to_ordinal(two)

three = np.array((
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (0, 2, 1),
    (0, 1, 0)
    ))

three_sorted = cast_to_ordinal(three)

four = np.array((
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (1, 2, 1),
    (2, 1, 1)
    ))

four_sorted = cast_to_ordinal(four)
#
primes = np.array((2, 3, 5))


make_plot(one, primes, currentdir + '/one', origin=True, legend=False,
          connect_color = 'black', dot_size=2, transparent=True, elev=13,
          azim=-58)
make_plot(one_sorted, primes, currentdir + '/one_sorted', origin=True, legend=False,
          connect_color = 'black', dot_size=2, transparent=True, elev=13,
          azim=-58)

make_2d_plot(two, currentdir + '/two')
make_2d_plot(two_sorted, currentdir + '/two_sorted')

make_plot(three, primes, currentdir + '/three', origin=True, legend=False,
          connect_color = 'black', dot_size=2, transparent=True, elev=13,
          azim=-58)
make_plot(three_sorted, primes, currentdir + '/three_sorted', origin=True, legend=False,
          connect_color = 'black', dot_size=2, transparent=True, elev=13,
          azim=-58)

make_plot(four, primes, currentdir + '/four', origin=True, legend=False,
          connect_color = 'black', dot_size=2, transparent=True, elev=13,
          azim=-58)
make_plot(four_sorted, primes, currentdir + '/four_sorted', origin=True, legend=False,
          connect_color = 'black', dot_size=2, transparent=True, elev=13,
          azim=-58)
# tex_matrix_writer(A, currentdir+'/A.tex')
# os.system('pdflatex -output-dir=./equations/ equations/A.tex')
# os.remove(currentdir + '/A' + '.aux')
# os.remove(currentdir + '/A' + '.log')
#
# tex_matrix_writer(A_sorted, currentdir+'/A_sorted.tex')
# os.system('pdflatex -output-dir=./equations/ equations/A_sorted.tex')
# os.remove(currentdir + '/A_sorted' + '.aux')
# os.remove(currentdir + '/A_sorted' + '.log')
#
# tex_matrix_writer(B, currentdir+'/B.tex')
# os.system('pdflatex -output-dir=./equations/ equations/B.tex')
# os.remove(currentdir + '/B' + '.aux')
# os.remove(currentdir + '/B' + '.log')
#
# tex_matrix_writer(B_sorted, currentdir+'/B_sorted.tex')
# os.system('pdflatex -output-dir=./equations/ equations/B_sorted.tex')
# os.remove(currentdir + '/B_sorted' + '.aux')
# os.remove(currentdir + '/B_sorted' + '.log')
#
# tex_matrix_writer(C, currentdir+'/C.tex')
# os.system('pdflatex -output-dir=./equations/ equations/C.tex')
# os.remove(currentdir + '/C' + '.aux')
# os.remove(currentdir + '/C' + '.log')
#
# tex_matrix_writer(C_sorted, currentdir+'/C_sorted.tex')
# os.system('pdflatex -output-dir=./equations/ equations/C_sorted.tex')
# os.remove(currentdir + '/C_sorted' + '.aux')
# os.remove(currentdir + '/C_sorted' + '.log')
#
# tex_matrix_writer(D, currentdir+'/D.tex')
# os.system('pdflatex -output-dir=./equations/ equations/D.tex')
# os.remove(currentdir + '/D' + '.aux')
# os.remove(currentdir + '/D' + '.log')
#
# print('')
# tex_matrix_writer(D_sorted, currentdir+'/D_sorted.tex')
# os.system('pdflatex -output-dir=./equations/ equations/D_sorted.tex')
# os.remove(currentdir + '/D_sorted' + '.aux')
# os.remove(currentdir + '/D_sorted' + '.log')
