import os

file_names = ['negative_tex', 'positive_tex', 'transposed_tex']
for fn in file_names:
    os.system('pdflatex -output-dir=./figure_6 figure_6/' + fn + '.tex')
    os.remove('figure_6/' + fn + '.aux')
    os.remove('figure_6/' + fn + '.log')
