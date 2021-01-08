import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0,parentdir)
from plot import *
from utils import get_layer_skew, get_layers



points = np.array((
(0, 0, 1),
(1, 0, 0),
(1, 0, 1),
(1, 1, 1),
(2, 1, 1),
(1, 1, 2),
(1, 2, 1),
(0, 1, 1),
(2, 1, 0),
(2, 2, 0)
))

layers = get_layers(points)
layer_skew = get_layer_skew(layers)
for i in range(len(layers)):
    print(layers[i])
    print(layer_skew[i])
    print()

make_plot(points, currentdir + '/slice_plot', layer_slices=range(1, 5), connect_color='black',
          opacity=1, legend=False, origin=True, origin_range=[0, 5], azim=-56, 
          layer_color='lightgrey')

plot_ternary(points, currentdir + '/ternary_plot')
