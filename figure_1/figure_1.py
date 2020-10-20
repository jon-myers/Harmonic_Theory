from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import itertools
from fractions import Fraction
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# def get_connections(pts):
#     combs = list(itertools.combinations(range(len(pts))))
#     combs = [i for i in combs if np.abs(np.sum(pts[i[0]] - pts[i[1]])) == 1]
#     print(combs)
#     print(pts)
    

#grids
bounds = [0, 2]
points = np.arange(bounds[0], bounds[1]+1)
fig = plt.figure(figsize=[8, 8])
ax = mplot3d.Axes3D(fig, elev=16, azim=-72)

primes = np.array((2.0, 3.0, 5.0))

pts = np.array([
[0, 0, 0], 
[1, 0, 0],
[2, 0, 0], 
[0, 1, 0],
[1, 1, 0],
[2, 1, 0], 
[0, 0, 1], 
[1, 0, 1],
[2, 0, 1], 
[0, 1, 1], 
[1, 1, 1],
[2, 1, 1]
])
pts += [-1, 0, 0]


def get_ratios(pts, primes):
    prods = np.product(primes ** pts, axis=1)
    fracs = [str(Fraction(i).numerator) + ':' + str(Fraction(i).denominator) for i in prods]
    return fracs
    

segments = np.array([
[[0, 2], [0, 0], [0, 0]],
[[0, 2], [0, 0], [1, 1]],
[[0, 2], [1, 1], [0, 0]],
[[0, 2], [1, 1], [1, 1]],
[[0, 0], [0, 1], [0, 0]],
[[1, 1], [0, 1], [0, 0]],
[[2, 2], [0, 1], [0, 0]],
[[0, 0], [0, 1], [1, 1]],
[[1, 1], [0, 1], [1, 1]],
[[2, 2], [0, 1], [1, 1]],
[[0, 0], [0, 0], [0, 1]],
[[1, 1], [0, 0], [0, 1]],
[[2, 2], [0, 0], [0, 1]],
[[0, 0], [1, 1], [0, 1]],
[[1, 1], [1, 1], [0, 1]],
[[2, 2], [1, 1], [0, 1]]
])
segments += np.array([[-1, -1], [0, 0], [0, 0]])

ratios = get_ratios(pts, primes)
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='black', depthshade=False, s=60)
for i, pt in enumerate(pts):
    ax.text(pt[0] - 0.05, pt[1] + 0.1, pt[2], ratios[i], color='black', horizontalalignment='right')
for seg in segments:
    ax.plot(seg[0], seg[1], seg[2], color='grey', alpha=0.5)

x_arrow = Arrow3D([1.5, 1.75], [0, 0], [1, 1], mutation_scale=20, lw=3, arrowstyle='-|>', color='black')
y_arrow = Arrow3D([1.5, 1.5], [0, 0.75], [1, 1], mutation_scale=20, lw=3, arrowstyle='-|>', color='black')
z_arrow = Arrow3D([1.5, 1.5], [0, 0], [1, 1.35], mutation_scale=20, lw=3, arrowstyle='-|>', color='black')



ax.text(1.725, 0, 0.9, '2', color='black', size='large')
ax.text(1.5, 0.75, 0.925, '3', color='black', size='large')
ax.text(1.53, 0, 1.3, '5', color='black', size='large')

    
ax.set_axis_off()
ax.set_xlim3d([-1, 1])
ax.set_ylim3d([0, 2])
ax.set_zlim3d([0, 2])
# set_axes_equal(ax)
plt.savefig('figure_1.png')
