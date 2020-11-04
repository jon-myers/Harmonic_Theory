import itertools
from fractions import Fraction
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import math
import matplotlib.colors

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

def hz_to_cents(hz, root):
    return 1200 * math.log2(hz / root)

def get_segments(pts):
    combs = list(itertools.combinations(range(len(pts)), 2))
    segments = np.array([(pts[i[0]], pts[i[1]]) for i in combs])
    is_neighbor = np.sum(np.abs(segments[:, 0] - segments[:, 1]), axis=1) == 1
    segments = segments[is_neighbor]
    segments = np.transpose(segments, axes=(0, 2, 1))
    return segments

def get_ratios(pts, primes, octaves = None, oct_generalized = False, string=True):
    if np.all(octaves == None):
        octaves = np.repeat(0, len(primes))
    prods = np.product((primes * (2.0**octaves)) ** pts, axis=1)
    fracs = [Fraction(i).limit_denominator(1000) for i in prods]
    if oct_generalized == True:
        for f in range(len(fracs)):
            while fracs[f] >= 2:
                fracs[f] /= 2
            while fracs[f] < 1:
                fracs[f] *= 2
    str_fracs = [str(i.numerator) + ':' + str(i.denominator) for i in fracs]
    if string == True:
        return str_fracs
    else:
        return fracs

def make_plot(pts, primes, path, octaves = None, draw_points = None,
              oct_generalized = False, dot_size=1, colors=None, ratios=True,
              origin=False, origin_range = [-2, 3], get_ax=False, legend=True,
              range_override=[0, 0], transparent=False, connect_color='grey'):
    c = matplotlib.colors.get_named_colors_mapping()
    if np.all(colors == None):
        colors = ['black' for i in range(len(pts))]
    else:
        colors = [c[i.lower()] for i in colors]
    if np.all(octaves == None):
        octaves = np.repeat(0, len(primes))
    if np.all(draw_points == None):
        segments = get_segments(pts)
    else:
        segments = get_segments(np.concatenate((pts, draw_points)))
    ratios = get_ratios(pts, primes, octaves, oct_generalized)
    fig = plt.figure(figsize=[8, 6])
    ax = mplot3d.Axes3D(fig, elev=16, azim=-72)
    ax.set_axis_off()

    min = np.min(pts)
    if min > range_override[0]:
        min = range_override[0]
    max = np.max(pts)
    if max < range_override[1]:
        max = range_override[1]


    if origin == True:
        quiver_min = origin_range[0]
        quiver_max = origin_range[1]
        q_diff = quiver_max - quiver_min
        if max < quiver_max:
            max = quiver_max
        if min > quiver_min:
            min = quiver_min
        x, y, z = np.array([[quiver_min, 0, 0],[0, quiver_min, 0],[0, 0, quiver_min]])
        u, v, w = np.array([[q_diff, 0, 0],[0, q_diff, 0],[0, 0, q_diff]])
        ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black", zorder=-1)
        for tick in range(quiver_min+1, quiver_max):
            a = [-0.0625, 0.0625]
            b = [0, 0]
            c = [tick, tick]
            ax.plot(a, b, c, color='black')
            ax.plot(b, a, c, color='black')
            ax.plot(a, c, b, color='black')
            ax.plot(c, a, b, color='black')
            ax.plot(b, c, a, color='black')
            ax.plot(c, b, a, color='black')

    xyz = [pts[:, 0], pts[:, 1], pts[:, 2]]
    for i, pt in enumerate(pts):
        ax.scatter(pt[0], pt[1], pt[2], color=colors[i], depthshade=False,
                   s=int(60 * dot_size))
    # ax.scatter(*xyz, color=colors, depthshade=False, s=int(60 * dot_size))
    if ratios == True:
        for i, pt in enumerate(pts):
            ax.text(pt[0] - 0.15, pt[1] + 0.25, pt[2], ratios[i], c='black',
                    horizontalalignment='right', size='large')
    for seg in segments:
        ax.plot(seg[0], seg[1], seg[2], color=connect_color, alpha=0.5, lw=0.5*dot_size)

    ax.set_xlim3d([min, max])
    ax.set_ylim3d([min, max])
    ax.set_zlim3d([min, max])
    plt.savefig(path + '.pdf', transparent=transparent)
    plt.close(fig)

    if legend == True:

        fig = plt.figure(figsize=[8, 6])
        ax = mplot3d.Axes3D(fig, elev=16, azim=-72)
        x_arrow = Arrow3D([-.005, 0.25], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle='-|>', color='black')
        y_arrow = Arrow3D([0, 0], [-0.01, 0.5], [0, 0], mutation_scale=20, lw=1, arrowstyle='-|>', color='black')
        z_arrow = Arrow3D([0, 0], [0, 0], [-0.01, 0.35], mutation_scale=20, lw=1, arrowstyle='-|>', color='black')
        ax.add_artist(x_arrow)
        ax.add_artist(y_arrow)
        ax.add_artist(z_arrow)
        ratios = primes * (2.0 ** octaves)

        ax.text(0.25, 0, 0, Fraction(ratios[0]), color='black', size=4 * dot_size)
        ax.text(0, 0.5, -0.03, Fraction(ratios[1]), color='black', size=4 * dot_size)
        z = ax.text(0.03, 0, 0.3, Fraction(ratios[2]), color='black', size=4 * dot_size)
        ax.set_axis_off()
        plt.savefig(path + '_legend.pdf', transparent=transparent)
        plt.close()




def make_shell_plot(shell, pts, primes, path, octaves = None, draw_points = None,
              oct_generalized = False, dot_size=1, colors=None, ratios=True,
              origin=False, origin_range = [-2, 3], get_ax=False, legend=True,
              range_override=[0, 0], transparent=False, shell_color='grey',
              point_color='black'):
    c = matplotlib.colors.get_named_colors_mapping()
    if np.all(colors == None):
        colors = ['black' for i in range(len(pts))]
    else:
        colors = [c[i.lower()] for i in colors]
    if np.all(octaves == None):
        octaves = np.repeat(0, len(primes))
    if np.all(draw_points == None):
        point_segments = get_segments(pts)
        shell_segments = get_segments(shell)
    else:
        point_segments = get_segments(np.concatenate((pts, draw_points)))
        shell_segments = get_segments(shell)
    ratios = get_ratios(pts, primes, octaves, oct_generalized)
    fig = plt.figure(figsize=[8, 6])
    ax = mplot3d.Axes3D(fig, elev=16, azim=-72)
    ax.set_axis_off()

    min = np.min(shell)
    if min > range_override[0]:
        min = range_override[0]
    max = np.max(shell)
    if max < range_override[1]:
        max = range_override[1]


    if origin == True:
        quiver_min = origin_range[0]
        quiver_max = origin_range[1]
        q_diff = quiver_max - quiver_min
        if max < quiver_max:
            max = quiver_max
        if min > quiver_min:
            min = quiver_min
        x, y, z = np.array([[quiver_min, 0, 0],[0, quiver_min, 0],[0, 0, quiver_min]])
        u, v, w = np.array([[q_diff, 0, 0],[0, q_diff, 0],[0, 0, q_diff]])
        ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color="black", zorder=-1)
        for tick in range(quiver_min+1, quiver_max):
            a = [-0.0625, 0.0625]
            b = [0, 0]
            c = [tick, tick]
            ax.plot(a, b, c, color='black')
            ax.plot(b, a, c, color='black')
            ax.plot(a, c, b, color='black')
            ax.plot(c, a, b, color='black')
            ax.plot(b, c, a, color='black')
            ax.plot(c, b, a, color='black')

    for i, pt in enumerate(shell):
        ax.scatter(pt[0], pt[1], pt[2], color=c[shell_color], depthshade=False,
                   s=int(60 * dot_size))
    for seg in shell_segments:
        ax.plot(seg[0], seg[1], seg[2], color=c[shell_color], alpha=0.5, lw=0.5*dot_size)

    for i, pt in enumerate(pts):
        if i == 0:
            ax.scatter(pt[0], pt[1], pt[2], color='red', depthshade=False,
                       s=int(60 * dot_size))
        else:
            ax.scatter(pt[0], pt[1], pt[2], color=c[point_color], 
                       depthshade=False, s=int(60 * dot_size))
    for seg in point_segments:
        ax.plot(seg[0], seg[1], seg[2], color=c[point_color], alpha=0.5, lw=0.5*dot_size)

    if ratios == True:
        for i, pt in enumerate(pts):
            ax.text(pt[0] - 0.15, pt[1] + 0.25, pt[2], ratios[i], c='black',
                    horizontalalignment='right', size='large')


    ax.set_xlim3d([min, max])
    ax.set_ylim3d([min, max])
    ax.set_zlim3d([min, max])
    plt.savefig(path + '.pdf', transparent=transparent)
    plt.close(fig)

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def is_fully_connected(points):
    """Returns True if all points are exactly one unit away from at least one
    other point; else returns False"""
    full_truth = []
    for point in points:
        other_points = points[np.invert((points == point).all(axis=1))]
        truth_array = []
        for op in other_points:
            one = np.count_nonzero(np.abs(point - op) == 1) == 1
            zero = np.count_nonzero(point - op == 0) == 2
            truth = one and zero
            truth_array.append(truth)
        full_truth.append(np.any(truth_array))
    return np.all(full_truth)


def sub_branches(points):
    """Given the points that make up a chord, returns the list of all subsets of
    those points that form branches."""
    out = []
    size = len(points)
    while True:
        potential_indexes = np.array([i for i in itertools.combinations(range(len(points)), size)])
        for pi in potential_indexes:
            if is_fully_connected(points[pi]):
                out.append(pi)
        size -= 1
        if size == 1:
            break
    return [points[i] for i in out]

def get_transposition_shell(points):
    """Given the points that make up a branch rooted at the origin, returns the
    points that make up its rotation shell."""
    dims = np.shape(points)[-1]
    permutations = np.array([i for i in itertools.permutations(range(dims))])
    perms = points[:, permutations]
    transpositions = perms.transpose((1, 0, *range(2, len(np.shape(perms)))))
    all_points = np.concatenate(transpositions)
    unique = np.unique(all_points, axis=0)
    return unique

def is_contained_by(point, container):
    """Returns True if point is contained by container"""
    return np.all(point - container >= 0)

def are_roots(points):
    """Returns an array of boolean values assessing if each point is a root by
    testing if each point is contained by any of the other points."""
    out = []
    for point in points:
        other_points = points[np.invert((points == point).all(axis=1))]
        truth_array = []
        for op in other_points:
            truth_array.append(is_contained_by(point, op))
        out.append(not np.any(np.array(truth_array)))
    return np.array(out)

def unique_permutations(arr):
    return np.array(list(set(itertools.permutations(arr))))
    
def paths_to_point(point, root = [0, 0, 0]):
    """Returns a list of sets of points outlining each of the possible paths 
    between root and point."""
    point = np.array(point)
    root = np.array(root)
    point = point - root
    num_of_paths = np.product(point + 1)
    increments_0 = np.repeat(0, point[0])
    increments_1 = np.repeat(1, point[1])
    increments_2 = np.repeat(2, point[2])
    increments = np.hstack((increments_0, increments_1, increments_2))   
    inc_paths = unique_permutations(increments)
    paths = [] 
    for ip in inc_paths:
        path = np.zeros((len(ip) + 1, 3), dtype = int)
        for i, item in enumerate(ip):
            path[i+1:, item] += 1
        path += root
        paths.append(path)
    paths = np.array(paths)
    return paths
    
