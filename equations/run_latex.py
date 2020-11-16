import os

file_names = ['rl_avg_dist']
for fn in file_names:
    os.system('pdflatex -output-dir=./equations equations/' + fn + '.tex')
    os.remove('equations/' + fn + '.aux')
    os.remove('equations/' + fn + '.log')
