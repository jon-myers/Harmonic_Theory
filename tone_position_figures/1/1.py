import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from plot import *
from utils import cast_to_ordinal

points = np.array((
(2, 0, 0),
(2, 0, 1),
(3, 0, 1),
(2, 1, 1),
(2, 1, 2), 
(1, 1, 2), 
(0, 1, 2),
(0, 1, 3),
(0, 2, 2)
))


points = cast_to_ordinal(points)
primes=[2, 3, 5]
make_plot(points, primes, currentdir + '/hsl', legend=False, azim=-75, 
          origin=True, connect_color='black', type='root_extremity', 
          labels=True, label_offset=[-0.18, 0.15, 0], origin_range=[0, 3], 
          dot_size=2, opacity=1)
plot_tree(points, currentdir + '/tree', type='root_extremity', labels=True)
