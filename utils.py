import itertools
from fractions import Fraction
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def cents_to_hz(cents, root):
    return root * (2 ** (1/12)) ** (cents/100)


def get_segments(pts):
    combs = list(itertools.combinations(range(len(pts)), 2))
    segments = np.array([(pts[i[0]], pts[i[1]]) for i in combs])
    is_neighbor = np.sum(np.abs(segments[:, 0] - segments[:, 1]), axis=1) == 1
    segments = segments[is_neighbor]
    segments = np.transpose(segments, axes=(0, 2, 1))
    return segments

def get_ratios(pts, primes, octaves = None):
    if np.all(octaves == None):
        octaves = np.repeat(0, len(primes))
    print(primes * (2.0**octaves))
    prods = np.product((primes * (2.0**octaves)) ** pts, axis=1)
    fracs = [Fraction(i).limit_denominator(1000) for i in prods]
    fracs = [str(i.numerator) + ':' + str(i.denominator) for i in fracs]
    return fracs

def make_plot(pts, primes, path, octaves = None, draw_points = None):
    if np.all(octaves == None):
        octaves = np.repeat(0, len(primes))
    if np.all(draw_points == None):
        segments = get_segments(pts)
    else:
        segments = get_segments(np.concatenate((pts, draw_points)))
    ratios = get_ratios(pts, primes, octaves)
    fig = plt.figure(figsize=[8, 6])
    ax = mplot3d.Axes3D(fig, elev=16, azim=-72)
    xyz = [pts[:, 0], pts[:, 1], pts[:, 2]]
    ax.scatter(*xyz, color='black', depthshade=False, s=60)
    for i, pt in enumerate(pts):
        ax.text(pt[0] - 0.05, pt[1] + 0.1, pt[2], ratios[i], c='black',
                horizontalalignment='right')
    for seg in segments:
        ax.plot(seg[0], seg[1], seg[2], color='grey', alpha=0.5)

    ax.set_axis_off()
    max = np.max(pts)
    min = np.min(pts)
    ax.set_xlim3d([min, max])
    ax.set_ylim3d([min, max])
    ax.set_zlim3d([min, max])
    plt.savefig(path + '.pdf')
    plt.close(fig)

    fig = plt.figure(figsize=[8, 6])
    ax = mplot3d.Axes3D(fig, elev=16, azim=-72)
    x_arrow = Arrow3D([-.005, 0.25], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle='-|>', color='black')
    y_arrow = Arrow3D([0, 0], [-0.01, 0.5], [0, 0], mutation_scale=20, lw=1, arrowstyle='-|>', color='black')
    z_arrow = Arrow3D([0, 0], [0, 0], [-0.01, 0.35], mutation_scale=20, lw=1, arrowstyle='-|>', color='black')
    ax.add_artist(x_arrow)
    ax.add_artist(y_arrow)
    ax.add_artist(z_arrow)
    ratios = primes * (2.0 ** octaves)

    ax.text(0.25, 0, 0, Fraction(ratios[0]), color='black', size='large')
    ax.text(0, 0.5, -0.03, Fraction(ratios[1]), color='black', size='large')
    ax.text(0.03, 0, 0.3, Fraction(ratios[2]), color='black', size='large')
    ax.set_axis_off()
    plt.savefig(path + '_legend.pdf')
