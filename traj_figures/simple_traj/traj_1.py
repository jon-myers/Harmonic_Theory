import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)

from plot import plot_simple_trajectory
import numpy as np
from utils import tex_matrix_writer

traj = np.array((
(1, 0, 0),
(0, 1, 0),
(-1, 0, 0),
(0, 0, -1),
(-1, 0, 0),
(0, 0, 1),
(0, 0, 1)
))

plot_simple_trajectory(traj, currentdir + '/traj_1')
tex_path = currentdir + '/traj_1_tex'
tex_matrix_writer(traj, tex_path + '.tex')
os.system('pdflatex -output-dir=./traj_figures/simple_traj/ traj_figures/simple_traj/traj_1_tex.tex')
os.remove(tex_path + '.aux')
os.remove(tex_path + '.log')
