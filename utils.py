import itertools, json, math
from fractions import Fraction
import numpy as np
import math
import numpy_indexed as npi
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib.colors
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial.distance import cdist


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
    if len(segments) > 0:
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


# TODO remove plotting from utils and change import refs to import from plot.py
def make_plot(pts, primes, path, octaves = None, draw_points = None,
              oct_generalized = False, dot_size=1, colors=None, ratios=True,
              origin=False, origin_range = [-2, 3], get_ax=False, legend=True,
              range_override=[0, 0], transparent=False, connect_color='grey',
              draw_point_visible=False, draw_color='seagreen', connect_size=1,
              file_type='pdf', opacity=0.5, root_layout=False, elev=16, azim=-72):


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
    ax = mplot3d.Axes3D(fig, elev=elev, azim=azim)
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


    if root_layout == True:
        for pt in pts:
            ax.plot([0, pt[0]], [0, pt[1]], [0, pt[2]], color='green')
            ax.plot([pt[0], pt[0]], [0, pt[1]], [0, pt[2]], color='green')
            ax.plot([0, pt[0]], [pt[1], pt[1]], [0, pt[2]], color='green')
            ax.plot([0, pt[0]], [0, pt[1]], [pt[2], pt[2]], color='green')

    xyz = [pts[:, 0], pts[:, 1], pts[:, 2]]
    for i, pt in enumerate(pts):
        ax.scatter(pt[0], pt[1], pt[2], color=colors[i], depthshade=False,
                   s=int(60 * dot_size))
    if draw_point_visible==True:
        for i, pt in enumerate(draw_points):
            ax.scatter(pt[0], pt[1], pt[2], color=c[draw_color],
                       depthshade=False, s=int(60 * dot_size))
    if ratios == True:
        for i, pt in enumerate(pts):
            ax.text(pt[0] - 0.15, pt[1] + 0.25, pt[2], ratios[i], c='black',
                    horizontalalignment='right', size='large')
    for seg in segments:
        ax.plot(seg[0], seg[1], seg[2], color=connect_color, alpha=opacity, lw=connect_size)

    ax.set_xlim3d([min, max])
    ax.set_ylim3d([min, max])
    ax.set_zlim3d([min, max])
    plt.savefig(path + '.' + file_type, transparent=transparent)
    plt.close(fig)

    if legend == True:

        fig = plt.figure(figsize=[8, 6])
        ax = mplot3d.Axes3D(fig, elev=elev, azim=azim)
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
        plt.savefig(path + '_legend.' + file_type, transparent=transparent)
        plt.close()





def make_shell_plot(shell, pts, primes, path, octaves = None, draw_points = None,
              oct_generalized = False, dot_size=1, colors=None, ratios=True,
              origin=False, origin_range = [-2, 3], get_ax=False, legend=True,
              range_override=[0, 0], transparent=False, shell_color='grey',
              point_color='black', draw_point_visible=False, connect_size=1,
              shell_dot_size=None, angles=True):
    if shell_dot_size == None:
        shell_dot_size = dot_size
    c_ = matplotlib.colors.get_named_colors_mapping()
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
    ax = mplot3d.Axes3D(fig, elev=13, azim=-52)
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

    if angles == True and len(pts) > 1:
        combs = [i for i in itertools.combinations(range(len(pts)), 2)]
        for i, indices in enumerate(combs):
            arc = draw_arc(pts[indices[0]], pts[indices[1]], 0.2 + 0.05 *i)
            ax.scatter(arc[:,0], arc[:, 1], arc[:, 2], color='green', s=1)
        for pt in pts:
            ax.plot([0, pt[0]], [0, pt[1]], [0, pt[2]], color='blue', lw=connect_size)

    for i, pt in enumerate(shell):

        ax.scatter(pt[0], pt[1], pt[2], color=c_[shell_color], depthshade=False,
                   s=int(60 * shell_dot_size))
    for seg in shell_segments:
        ax.plot(seg[0], seg[1], seg[2], color=c_[shell_color], alpha=0.5, lw=connect_size)

    for i, pt in enumerate(pts):
        if i == 0:
            # spreviously color here was hard coded to red
            ax.scatter(pt[0], pt[1], pt[2], color=c_[point_color], depthshade=False,
                       s=int(60 * dot_size))
        else:
            ax.scatter(pt[0], pt[1], pt[2], color=c_[point_color],
                       depthshade=False, s=int(60 * dot_size))

    for seg in point_segments:
        ax.plot(seg[0], seg[1], seg[2], color=c_[point_color], alpha=0.5, lw=10)

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

def flatten(iterable):
    return list(itertools.chain.from_iterable(iterable))

def sub_branches(points):
    """Given the points that make up a chord, returns the list of all subsets of
    those points that form branches. If lens == true, the returned array is
    nested into groups by chord size, in decreasing order.
    """
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
    sb = [points[i] for i in out]
    return sb

def unique_sub_branches(points, count=False):
    """If given points, gets sub_branches, transfers all to ordinal, splits
    into groups based on length, and remove duplicates from each of those groups,
    before putting all unique sub branches back into an output array that is
    returned"""
    # TODO make this avoid getting the sub branches twice, by letting the input
    # be sub_branches or points.
    sb = [reorder_points(cast_to_ordinal(i)) for i in sub_branches(points)]
    lens = list(set([len(i) for i in sb]))[::-1]
    if count == True:
        sb_groups = [np.array([i for i in sb if len(i) == j]) for j in lens]
        usb_groups = [npi.unique(i, return_index=True, return_count=True) for i in sb_groups]
        usb = list(i[0] for i in usb_groups)
        idx = list(i[1] for i in usb_groups)
        cts = list(i[2] for i in usb_groups)

        all_matches = []
        ct = 0
        for l_i in range(len(sb_groups)):
            for unq in usb[l_i]:
                matches = []
                for i, sb in enumerate(sb_groups[l_i]):
                    intersect = npi.intersection(unq, sb)
                    if len(intersect) == len(unq):
                        matches.append(ct + i)
                all_matches.append(matches)
            ct += len(sb_groups[l_i])
        return flatten(usb), all_matches
    else:
        usb_groups = [npi.unique(np.array([i for i in sb if len(i) == j])) for j in lens]
        usb = flatten(usb_groups)
        return usb

def get_transposition_shell(points):
    """Given the points that make up a branch rooted at the origin, returns the
    points that make up its rotation shell."""
    dims = np.shape(points)[-1]
    permutations = np.array([i for i in itertools.permutations(range(dims))])
    # print(np.shape(points))
    # print(np.shape(permutations))
    # try:
    #     perms = points[:, permutations]
    # except:
    #     print(points, permutations)
    #     perms = points[0][:, permutations]
    perms = points[:, permutations]
    transpositions = perms.transpose((1, 0, *range(2, len(np.shape(perms)))))
    all_points = np.concatenate(transpositions)
    unique = np.unique(all_points, axis=0)
    return unique

def get_multipath_shell(points):
    for point in points:
        multipath = cartesian_product(*[np.arange(i+1) for i in point])
        if 'multipath_shell' not in locals():
            multipath_shell = multipath
        else:
            multipath_shell = np.vstack((multipath_shell, multipath))
    return np.unique(multipath_shell, axis=0)

def get_transpositions(points):
    points = np.array(points)
    dims = np.shape(points)[-1]
    permutations = np.array([i for i in itertools.permutations(range(dims))])
    perms = points[:, permutations]
    transpositions = perms.transpose((1, 0, *range(2, len(np.shape(perms)))))
    return transpositions

def get_stability(points):
        """The average of the proportion of rotations in which each unique
        position is occupied"""
        rots = get_transpositions(points)
        shape = np.shape(rots)
        pos_in_rots = np.unique(rots, axis=1)
        pos_tot = npi.union(*[i for i in rots])
        all_pos_occurences = pos_in_rots.reshape((np.int(np.size(pos_in_rots) / 3), 3))
        pos_tot, counts = np.unique(all_pos_occurences, axis=0, return_counts=True)
        out = np.round(np.mean(counts) / 6, 2)
        return out

def get_loops(points):
    """Returns the number of loops in the structure of the chord"""
    ct = 0
    diffs = points[:, None] - points[None, :]
    diffs = np.linalg.norm(diffs, axis=2)

    one_inds = np.array(np.nonzero(diffs == 1)).T
    root_two_inds = np.array(np.nonzero(diffs == np.sqrt(2))).T

    filtered_root_two_inds = []
    for i in root_two_inds:
        truth = True
        for j in filtered_root_two_inds:
            if np.all(i[::-1] == j):
                truth = False
        if truth:
            filtered_root_two_inds.append(i)
    filtered_root_two_inds = np.array(filtered_root_two_inds)

    for i in filtered_root_two_inds:
        ones_a = one_inds[np.nonzero(one_inds[:,0] == i[0])]
        ones_b = one_inds[np.nonzero(one_inds[:,0] == i[1])]
        points_a = points[ones_a[:, 1]]
        points_b = points[ones_b[:, 1]]
        intersections = npi.intersection(points_a, points_b)
        if len(intersections) == 2:
            ct += 1
    return ct / 2

def get_complement_shell(points):
    """Given the points that make up a branch rooted at the origin, returns the
    points that make up its complement shell."""
    maxs = np.max(points, axis=0)
    complement_shell = np.array(((0, 0, 0))).reshape((1, 3))
    for point in points:
        shell = [np.array([k for k in range(i + 1)]) for i in point]
        shell = cartesian_product(*shell)
        complement_shell = np.concatenate((complement_shell, shell), axis=0)
    complement_shell = np.unique(complement_shell, axis=0)
    return complement_shell

def is_contained_by(point, container):
    """Returns True if point is contained by container"""
    return np.all(point - container >= 0)

def containment_relationship(A, B):
    """For two points, returns 0 if there is no containment relationship,
    1 if A contains B, and 2 if B contains A."""
    if is_contained_by(B, A):
        return 1
    elif is_contained_by(A, B):
        return 2
    else:
        return 0

def are_roots(points):
    """Returns an array of boolean values assessing if each point is a root by
    testing if each point is contained by any of the other points."""
    out = []
    points = np.array(points)
    for point in points:
        other_points = points[np.invert((points == point).all(axis=1))]
        truth_array = []
        for op in other_points:
            truth_array.append(is_contained_by(point, op))
        out.append(not np.any(np.array(truth_array)))
    return np.array(out)

def are_extremities(points):
    """Returns an array of boolean values assessing if each point is an
    extremity by testing if each point contains any other points."""
    out = []
    for point in points:
        other_points = points[np.invert((points == point).all(axis=1))]
        truth_array = []
        for op in other_points:
            truth_array.append(is_contained_by(op, point))
        out.append(not np.any(np.array(truth_array)))
    return np.array(out)

def are_root_breakpoints(points):
    """Returns an array of boolean values assessing if each point is a root
    breakpoint by testing if it is contained by two roots, and is the simplest
    point contained by those two roots"""
    points = np.array(points)
    roots = points[are_roots(points)]
    combs = itertools.combinations(range(len(roots)), 2)
    out = np.zeros(len(points), dtype=bool)
    for comb in combs:
        potential_breakpoints = []
        for point in points:
            test_1 = is_contained_by(point, roots[comb[0]])
            test_2 = is_contained_by(point, roots[comb[1]])
            if test_1 and test_2:
                potential_breakpoints.append(point)
        potential_breakpoints = np.array(potential_breakpoints)
        if len(potential_breakpoints) > 1:
            sorts = np.argsort([sum(i) for i in potential_breakpoints])
            breakpoint = potential_breakpoints[sorts][0]
        elif len(potential_breakpoints) == 1:
            breakpoint = potential_breakpoints[0]
        elif len(potential_breakpoints) == 0:
            breakpoint = None
        for i, pt in enumerate(points):
            if str(breakpoint) == str(pt):
                out[i] = True
    return out

def are_extremity_breakpoints(points):
    """Returns an array of boolean values assessing if each point is an extremity
    breakpoint by testing if it contains by two extremities, and is the most complex
    point that contains those two roots"""
    points = np.array(points)
    extremities = points[are_extremities(points)]
    combs = itertools.combinations(range(len(extremities)), 2)
    out = np.zeros(len(points), dtype=bool)
    for comb in combs:
        potential_breakpoints = []
        for point in points:
            test_1 = is_contained_by(extremities[comb[0]], point)
            test_2 = is_contained_by(extremities[comb[1]], point)
            if test_1 and test_2:
                potential_breakpoints.append(point)
        potential_breakpoints = np.array(potential_breakpoints)
        if len(potential_breakpoints) > 1:
            sorts = np.argsort([sum(i) for i in potential_breakpoints])
            breakpoint = potential_breakpoints[sorts][-1]
        elif len(potential_breakpoints) == 1:
            breakpoint = potential_breakpoints[0]
        elif len(potential_breakpoints) == 0:
            breakpoint = None
        for i, pt in enumerate(points):
            if str(breakpoint) == str(pt):
                out[i] = True
    return out

def unique_permutations(arr):
    return np.array(list(set(itertools.permutations(arr))))

def paths_to_point(point, root = [0, 0, 0]):
    """Returns a list of sets of points outlining each of the possible paths
    between root and point. (Currently only for 3D)"""
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

def cast_to_ordinal(points, verbose=False):
    # needs to be fixed, not discriminating enough!
    """Given a list of harmonic space vectors, returns the collection cast to
    ordinal position. That is, with dimensions sorted firstly by max extent from
    origin, secondly by average extent in each dimension, and thirdly by order
    of extent in each dimension of point with furthest manhattan
    distance from origin."""
    # TODO DOESN't work in more than 4 dimensions!!!

    mins = np.min(points, axis=0)
    points = points - mins
    origin = np.repeat(0, np.shape(points)[-1])
    dims = np.shape(points)[-1]

    avg = np.average(points, axis=0)
    avg_dup_indexes = indexes_of_duplicates(avg)
    if verbose and len(avg_dup_indexes) > 1:

        print('Potentially fatal: multiple avg dups!')
        print(points)
    if len(avg_dup_indexes) == 1:
        avg_dup_indexes = avg_dup_indexes[0]
    avg_order = np.argsort(-1 * avg)
    points = points[:, avg_order]

    maxs = np.max(points) - np.max(points, axis=0)
    avg_dup_maxs = indexes_of_duplicates(maxs)
    if verbose and len(avg_dup_maxs) > 1:
        print('Potentially fatal: multiple avg maxs!')
        print(points)
    if len(avg_dup_maxs) == 1:
        avg_dup_maxs = avg_dup_maxs[0]
    max_order = np.argsort(maxs)
    points = points[:, max_order]
    # print(avg_dup_indexes, avg_dup_maxs, '\n\n\n')
    shared_dims = np.intersect1d(avg_dup_indexes, avg_dup_maxs)
    if verbose and len(shared_dims) > 2:
        print('Potentially fatal: shared dims greater than 2!')
        print(points, '\n')
    if len(shared_dims) == 2 and dims > 2:
        dims = np.arange(np.shape(points)[-1])
        non_shared_dims = dims[np.invert(npi.contains(shared_dims, dims))]

        #discriminatory collection
        shared_dim_points = points[:, shared_dims]
        equal_filter = shared_dim_points[:,0] - shared_dim_points[:,1] == 0

        non_shared_dim_points = points[:, non_shared_dims]
        seperated_dup_inds = indexes_of_duplicates_2d(non_shared_dim_points)

        inverted_inds = []
        for dup_inds in seperated_dup_inds:
            combs = itertools.combinations(dup_inds, 2)
            for comb in combs:
                if np.all(shared_dim_points[comb[0]] == shared_dim_points[comb[1]][::-1]):
                    for c in comb:
                        inverted_inds.append(c)
        inverted_filter = npi.contains(np.array(inverted_inds), np.arange(len(points)))
        filter = np.invert(np.logical_or(equal_filter,inverted_filter))
        C_dis = points[filter]
        if len(C_dis > 0):
            mds = np.sum(C_dis, axis=1)
            max_mds = np.max(mds)
            # if np.count_nonzero(mds == max_mds) > 1:
            #     print('Potentially fatal: shared max manhattan distances in C_dis')
            #     print(mds)
            #     print(C_dis)
            T_mmd = C_dis[np.argmax(mds)]
            dis_sorts = np.argsort(T_mmd[shared_dims])[::-1]
            sorts = dims[:]
            sorts[shared_dims] = dis_sorts
            sorts = np.argsort(sorts)
            points = points[:, sorts]
    return points

def indexes_of_duplicates(arr):
    """Given a 1-d numpy array, arr, returns a list of numpy arrays each filled with
    the indexes of any items in arr that appear more than once."""
    arr = np.array(arr)
    idx_sort = np.argsort(arr)
    sorted = arr[idx_sort]
    vals, idx_start, count = np.unique(sorted, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:])
    vals = vals[count > 1]
    res = filter(lambda x: x.size > 1, res)
    return list(res)

def indexes_of_duplicates_2d(arr):
    """Given a 2-d numpy array, arr, returns a list of numpy arrays each filled with
    the indexes of any items in arr that appear more than once."""
    arr = np.array(arr)
    unq = npi.unique(arr)
    out = [np.nonzero(npi.contains([u], arr))[0] for u in unq]
    out = [i for i in out if len(i) > 1]
    return out

def get_ordinal_sorts(points):
    """Returns a sorting array that would cast a set of points to ordinal
    position. """
    mins = np.min(points, axis=0)
    points = points - mins
    origin = np.repeat(0, np.shape(points)[-1])

    # index of max manhattan distance
    bc_origin = np.broadcast_to(origin, np.shape(points))
    manhattan_distance = cdist(bc_origin, points, metric='cityblock')
    max_md_index = np.argmax(manhattan_distance)
    max_md_order = np.argsort(points[max_md_index])[::-1]
    points = points[:, max_md_order]

    avg_order = np.argsort(-1 * np.average(points, axis=0))
    points = points[:, avg_order]
    max_order = np.argsort(np.max(points) - np.max(points, axis=0))
    points = points[:, max_order]
    return np.array((max_md_order, avg_order, max_order))

def reorder_points(points):
    """Reorders points such that they are in a consistent order for testing for
    uniqueness against other sets of points."""
    primes = np.array((2.0, 3, 5, 7, 11, 13, 17, 19))[:np.shape(points)[-1]]
    mult = np.product(primes ** points, axis=1)
    indexes = np.argsort(mult)
    return points[indexes]

def draw_arc(A, B, r = 0.25):
    A = np.where(A != 0, A / np.linalg.norm(A), A)
    B = np.where(B != 0, B / np.linalg.norm(B), B)
    crossed = np.cross(A, B)
    B_alt = np.cross(crossed, A)

    B_alt = np.where(B_alt != 0, B_alt / np.linalg.norm(B_alt), B_alt)
    theta_limit = np.arccos(np.dot(A, B))


    theta = np.repeat(np.linspace(0, theta_limit, 100), 3).reshape((100, 3))
    return r * (np.cos(theta) * A + np.sin(theta) * B)

def create_tree_edges(points):
    # make a list of tuples describing orthogonal connections
    # and make a list of containments, for dotted lines
    out = []
    for p, point in enumerate(points):
        for op, other_point in enumerate(points[p:]):
            if np.linalg.norm(point - other_point) == 1:
                if is_contained_by(point, other_point):
                    out.append((p, op+p))
                else:
                    out.append((op+p, p))
    # out = sorted(out, key=lambda x: x[0])
    return out

def plot_tree(points, path, type='root'):
    """

    possible types: root, extremity, root_extremity, root_breakpoint,
    extremity_breakpoint,"""
    edges = create_tree_edges(points)
    G=nx.MultiDiGraph(size='2, 4')
    G.add_edges_from(edges)
    edge_order = []
    for i in itertools.chain.from_iterable(edges):
        if i not in edge_order: edge_order.append(i)
    edge_order = np.array(edge_order)
    colors = np.repeat(0, len(points))

    if type == 'root':
        colors = np.where(are_roots(points), 1, colors)
        colors = [['black', 'red'][i] for i in colors]
    elif type == 'extremity':
        colors = np.where(are_extremities(points), 1, colors)
        colors = [['black', 'mediumseagreen'][i] for i in colors]
    elif type == 'root_extremity':
        colors = np.where(are_roots(points), 1, colors)
        colors = np.where(are_extremities(points), 2, colors)
        colors = [['black', 'red', 'mediumseagreen'][i] for i in colors]
    elif type == 'root_breakpoint':
        colors = np.where(are_roots(points), 1, colors)
        colors = np.where(are_root_breakpoints(points), 2, colors)
        colors = [['black', 'red', 'mediumorchid'][i] for i in colors]
    elif type == 'extremity_breakpoint':
        colors = np.where(are_extremities(points), 1, colors)
        colors = np.where(are_extremity_breakpoints(points), 2, colors)
        colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in colors]

    colors = [colors[i] for i in edge_order]

    pos=graphviz_layout(G, prog='dot')
    plt.figure(figsize=[3, 4])
    nx.draw(G, pos, with_labels=False, arrows=False, node_color=colors)
    plt.savefig(path + '.pdf', transparent=True)
    plt.close()

def plot_basic_hsl(points, path, type='root'):
    """

    possible types: root, extremity, root_extremity, root_breakpoint,
    extremity_breakpoint,"""
    colors = np.repeat(0, len(points))
    if type == 'root':
        colors = np.where(are_roots(points), 1, colors)
        colors = [['black', 'red'][i] for i in colors]
    elif type == 'extremity':
        colors = np.where(are_extremities(points), 1, colors)
        colors = [['black', 'mediumseagreen'][i] for i in colors]
    elif type == 'root_extremity':
        colors = np.where(are_roots(points), 1, colors)
        colors = np.where(are_extremities(points), 2, colors)
        colors = [['black', 'red', 'mediumseagreen'][i] for i in colors]
    elif type == 'root_breakpoint':
        colors = np.where(are_roots(points), 1, colors)
        colors = np.where(are_root_breakpoints(points), 2, colors)
        colors = [['black', 'red', 'mediumorchid'][i] for i in colors]
    elif type == 'extremity_breakpoint':
        colors = np.where(are_extremities(points), 1, colors)
        colors = np.where(are_extremity_breakpoints(points), 2, colors)
        colors = [['black', 'mediumseagreen', 'cornflowerblue'][i] for i in colors]
    else: print('Error: Unknown Type')
    primes = np.array((3, 5, 7))
    make_plot(points, primes, path, dot_size=2, colors=colors,
              ratios=False, range_override=[-1, 3], connect_color='black',
              connect_size=1, legend=False, transparent=True)

def get_factors(nr):
    """Enumerates all factors of a given number."""
    i = 2
    factors = []
    while i <= nr:
        if (nr % i) == 0:
            factors.append(i)
            nr = nr / i
        else:
            i = i + 1
    return factors

def get_hsv(nr, num_of_primes = 8):
    """For a given number, returns its harmonic space vector."""
    primes = np.array((2, 3, 5, 7, 11, 13, 17, 19))[:num_of_primes]
    hsv = np.zeros_like(primes)
    factors = np.array(get_factors(nr))
    unique, exponents = np.unique(factors, return_counts=True)
    for i, item in enumerate(unique):
        index = np.where(primes == item)
        hsv[index] = exponents[i]
    return hsv

def analyze(ratios, root = 1):
    ratios = [Fraction(ratio).limit_denominator(1000) for ratio in ratios]
    full_hsvs = np.array([get_hsv(f.numerator) - get_hsv(f.denominator) for f in ratios])
    octave_generalized = full_hsvs[:, 1:]
    primes = np.array((3, 5, 7, 11, 13, 17, 19))
    filter = np.where(np.any(octave_generalized.T != np.zeros(len(ratios)), axis=1))
    chord_primes = primes[filter]
    hsvs = octave_generalized[:, filter]
    shape = np.shape(hsvs)
    hsvs = np.reshape(hsvs, (shape[0], shape[2]))
    octs = full_hsvs[:, 0]
    trials = cartesian_product(*[np.arange(-4, 5) for i in range(len(chord_primes))])
    b_trials = np.expand_dims(trials, 1)
    b_trials_shape = np.shape(b_trials)
    b_trials = np.broadcast_to(b_trials, (b_trials_shape[0], len(hsvs), b_trials_shape[-1]))
    sum = np.sum(b_trials * hsvs, axis=2)
    possible_trials = trials[np.all(sum == octs, axis=1)]
    if len(possible_trials) == 0:
        print('No Dice!')
        oct_shifts = 'Cant find good octave shifts'
    elif len(possible_trials) == 1:
        oct_shifts = possible_trials[0]
    elif len(possible_trials) > 1:
        oct_shifts = possible_trials[np.argmin(np.sum(np.abs(possible_trials), axis=1))]

    return chord_primes, hsvs, oct_shifts

# clasifiers

def containment_size(points, return_containment_index=True):
    combs = [i for i in itertools.combinations(range(len(points)), 2)]
    ct = 0
    for comb in combs:
        p1 = points[comb[0]]
        p2 = points[comb[1]]
        if is_contained_by(p1, p2) or is_contained_by(p2, p1):
            ct += 1
    if return_containment_index == False:
        return ct
    else:
        return ct, ct / len([i for i in combs])



def get_routes(points):
    """Return the number of routes in a chord."""
    # intersections + extremities + loops - 1
    counts=[]
    for pt in points:
        dists = points-pt
        count = np.count_nonzero(np.linalg.norm(points-pt, axis=1) == 1)
        counts.append(count)
    counts = np.array(counts)
    intersections = np.count_nonzero(counts > 2)
    extremities = np.count_nonzero(counts == 1)
    loops = get_loops(points)
    return intersections + extremities + loops - 1

def mean_root_distance(points):
    roots = points[are_roots(points)]
    combined_dist = 0
    for root in roots:

        # dist = cdist(np.zeros_like(root), root, metric='cityblock')
        dist = np.sum(root)
        combined_dist += dist
    return combined_dist / len(roots)

def mean_root_angle(points):
    roots = points[are_roots(points)]
    combs = [i for i in itertools.combinations(range(len(roots)), 2)]
    angle_sum = 0
    for comb in combs:
        A = roots[comb[0]]
        B = roots[comb[1]]
        dot_product = np.dot(A, B)
        mag_product = np.linalg.norm(A) * np.linalg.norm(B)
        angle_sum += np.arccos(dot_product/ mag_product)
    if len(roots) == 1:
        return angle_sum
    else:
        return angle_sum * 2 * math.factorial(len(roots) - 2) / math.factorial(len(roots))



# trajectory utils
# ________________

def ult_vector(trajectory):
    """('Ultimate Vector') Returns the vector from the origin to the destination of the
    trajectory."""
    return np.sum(trajectory, axis=0)

def traj_to_points(traj, unique=True, persistence=False):
    """Convert a trajectory to the set of unique points it crosses, were it to
    start at the origin."""
    origin = np.zeros(np.shape(traj)[-1], dtype=int)
    points = np.expand_dims(origin, 0)
    for step in traj:
        new_point = np.array([points[-1] + step])
        points = np.concatenate((points, new_point))

    unq, cts = npi.unique(points, return_count=True)
    if persistence == True:
        if unique == True:
            return unq, cts / np.sum(cts)
        else:
            filter = npi.indices(unq, points)
            return points, cts[filter]
    else:
        if unique == True:
            return npi.unique(points)
        else:
            return points

def traj_to_point_tuples(traj, root = None, offset = 0.1):
    """Returns an array of tuples of the start and endpoint of each traj vector,
    pasted end-to-front."""
    if np.all(root == None):
        root = np.zeros(np.shape(traj)[-1])
        steps = []
    for i, step in enumerate(traj):
        if i == 0:
            start = root
        else:
            start = np.ceil(steps[-1][1])
        end = start + step * (1 - offset)
        steps.append((start, end))
    return steps




def cast_traj_to_ordinal(traj):
    """Reorders the axes of the trajectory such that the chord it generates
    is in ordinal position, except for being translated due to the origin."""
    points = traj_to_points(traj)
    sorts = get_ordinal_sorts(points)
    traj = traj[:, sorts[0]]
    traj = traj[:, sorts[1]]
    traj = traj[:, sorts[2]]
    return traj

def get_directionality(trajectory):
    """Returns the proportion of steps that point in the same direction as the
    ultimate vector, summed along each axis."""
    uv = ult_vector(trajectory)
    mult = uv * trajectory
    return np.sum(np.sign(mult)) / len(mult)

def get_crossings(trajectory, return_counts=True):
    """Returns all points that are visited at least twice."""
    points = traj_to_points(trajectory, unique=False)
    unq, cts = npi.unique(points, return_count=True)
    if return_counts == False:
        return unq[cts>1]
    else:
        return unq[cts>1], cts[cts>1]

# def get_persistence(trajectory):
#     """Returns proportion of all points relative to total non-unique points
#     crossed. Basically trying to measure how often each point is visited,
#     relative to the entirety of the trajectory."""
#     points = traj_to_points(unique=False)
#     _, cts = npi.unique(points, return_count=True)


def traj_decomposition(traj, show_unq=True, show_proportion=True):
    """Returns the list of ordered subsets of the original trajectory, and (if
    required) the list of unique subsets of the trajectory."""
    traj_inds = [i for i in range(len(traj))]
    decomp_index = []
    for i, end in enumerate(range(1, len(traj))[::-1]):
        steps = i + 2
        for step in range(steps):
            decomp_index.append(np.array(traj_inds[step: step + end]))
    decomp = [traj[i] for i in decomp_index]
    unq = unq_traj_decomp(decomp)
    prop = len(unq) /len(decomp)
    if show_unq and show_proportion:
        return decomp, unq, prop
    elif show_unq:
        return decomp, unq
    elif show_proportion:
        return decomp, prop
    else:
        return decomp

def unq_traj_decomp(decomp):
    """Takes a decomp (list of subsets of trajectory) and returns the list of
    unique sub-trajectories that have been cast to ordinal."""
    ord = [cast_traj_to_ordinal(i) for i in decomp]
    lens = list(set([len(i) for i in ord]))
    sb_groups = [npi.unique(np.array([i for i in ord if len(i) == j])) for j in lens]
    out = [i for i in itertools.chain.from_iterable(sb_groups)]
    return out


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def tex_matrix_writer(matrix, path):
    out = '\documentclass{article}\n'\
          '\\begin{document} $\n'\
          '\\left[\n'\
          '\\begin{array}'
    horiz = '|'.join(['c' for i in range(len(matrix))])
    out += '{' + horiz + '}\n'
    for dim in range(np.shape(matrix)[-1]):
        out += ' & '.join([str(i) for i in matrix[:,dim]]) + ' \\\\\n'
    out += '\end{array}\n'\
           '\\right] $\n'\
           '\end{document}'
    with open(path, 'w') as file:
        file.write(out)


def max_possible_roots(size, dims):
    """Given the number of points in a collection and dimensions of harmonic
    space it occupies, returns the maximum possible number of roots (or,
    extremities), assuming dims is 2 or more."""
    layers = math.floor(math.log(1 - size * (1 - dims), dims))
    roots = dims ** layers
    pts_left = size - (1 - roots) / (1 - dims)
    add = pts_left - int(np.ceil(pts_left / dims))
    roots += add
    return roots

def possible_paths(A, B):
    """Given two points in the same dimensionality, where A contains B, return
    all of the possible paths from A to B (not including the endpoints)."""
    test = is_contained_by(B, A)
    if not test:
        print('error!!')
    diff = B - A
    moves = []
    for i, steps in enumerate(diff):
        for step in range(steps):
            moves.append(i)
    permutations = list(set(itertools.permutations(moves)))
    paths = []
    for perm in permutations:
        path = A.reshape((1, *np.shape(A)))
        for move in perm:
            next_point = path[-1] + np.eye(np.shape(A)[-1], dtype=int)[move]
            next_point = next_point.reshape((1, *np.shape(next_point)))
            path = np.concatenate((path, next_point))
        paths.append(path[1:-1])
    return paths


# possible_paths(np.array((0, 0, 0)), np.array((2, 1, 0)))

def get_connected_indexes(points):
    # first, are they all connected? if so, connected_indexes will have one
    # list in it, containing all tones. If not, connected indexes will have
    # a seperate list for each subset of tones that are connected via adjacency.
    indexes = np.arange(len(points))
    connected_indexes = []
    outer_ct = 0
    out_size = 0
    while out_size < np.shape(points)[0]:
        accounted_for = list(itertools.chain.from_iterable(connected_indexes))
        not_yet_accounted_for = indexes[np.invert(np.isin(indexes, accounted_for))]
        connected_indexes.append([not_yet_accounted_for[0]])
        ct = 0
        while ct < len(connected_indexes[outer_ct]):
            pt = points[connected_indexes[outer_ct][ct]]
            adjacencies = np.nonzero(np.sum(np.abs(pt - points), axis=1) == 1)[0]
            add_adjacencies = [i for i in adjacencies if i not in connected_indexes[outer_ct]]
            if len(add_adjacencies) > 0:
                for aa in add_adjacencies:
                    connected_indexes[outer_ct].append(aa)
            ct += 1
        outer_ct += 1
        out_size = len(list(i for i in itertools.chain.from_iterable(connected_indexes)))
    return connected_indexes


def least_common_containee(A, B):
    """For points A and B, return the simplest (lowest manhattan distance from
    origin) point that both points contain."""
    set = np.array((A, B))
    out = np.max(set, axis=0)
    return out

# test = np.array(((1, 0), (0, 1)))
# least_common_containee(*test)

def fix_collection(points):
    """For a collection of tones, assess if it is a 'neighborhood'—-that all
    points are connected by adjacency. If so, return the points and an array
    filled with ones, indicating that its a normal 'chord'. If not, finds
    'holes', adds them to the list of points, and returns an array with 1s for
    real tones, and 0s for 'holes', or 'ghost' tones."""

    connected_indexes = get_connected_indexes(points)

    if len(connected_indexes) == 1:
        return points, np.zeros(len(points)) + 1
    else:
        # check each subset of tones against the other subsets for containments.
        # If there are containments, add all possible path points as potential
        # holes.
        combs = itertools.combinations(np.arange(len(connected_indexes)), 2)
        potential_holes = []
        for comb in combs:
            n_A = points[connected_indexes[comb[0]]]
            n_B = points[connected_indexes[comb[1]]]
            trials = cartesian_product(np.arange(len(n_A)), np.arange(len(n_B)))
            crs = [containment_relationship(n_A[t[0]], n_B[t[1]]) for t in trials]
            for i, cr in enumerate(crs):
                if cr != 0:
                    if cr == 1:
                        start = n_A[trials[i][0]]
                        end = n_B[trials[i][1]]
                    if cr == 2:
                        start = n_B[trials[i][1]]
                        end = n_A[trials[i][0]]
                    pps = possible_paths(start, end)
                    pps_ = np.concatenate(pps)
                    if len(pps_) > 0:
                        if len(potential_holes) == 0:
                            potential_holes = pps_
                        else:
                            potential_holes = np.concatenate((potential_holes, pps_))
        unique, counts = npi.unique(potential_holes, return_count=True)
        sorted_unique = unique[np.argsort(counts)[::-1]]
        if len(sorted_unique) == 0:
            pot_holes = []
        else:
            pot_holes = sorted_unique[np.invert(npi.contains(points, sorted_unique))]

        # now try each 'pot_hole' (potential hole) to see if adding it to the
        # collection individually would bring all into one neighborhood. If none
        # can do alone, try every set of two, then every set of three, etc., all
        # the way up until you trying all pot holes altogether. If you can't
        # bring all together with the pot holes, bring together as many as
        # possible with as few as possible additions. From that point, you'll
        # have to find the simplest possible 'common hole' that would be contained
        # by the remaining sets ...
        pot_solutions = []
        for num_add_tones in range(1, len(pot_holes) + 1):
            if num_add_tones == 1:
                for i, ph in enumerate(pot_holes):
                    add_set = np.concatenate((points, np.expand_dims(ph, 0)))
                    ci = get_connected_indexes(add_set)
                    pot_solutions.append((np.array(i), len(ci)))
            else:
                combs = itertools.combinations(range(len(pot_holes)), num_add_tones)
                for comb in combs:
                    adds = pot_holes[np.array(comb)]
                    add_set = np.concatenate((points, adds))
                    ci = get_connected_indexes(add_set)
                    pot_solutions.append((comb, len(ci)))
        sorts = np.argsort([ps[1] for ps in pot_solutions], kind='stable')
        if len(pot_solutions) > 0:
            if pot_solutions[sorts[0]][1] == 1:
                add = np.array(pot_solutions[sorts[0]][0])
                add_pts = pot_holes[add]
                if add_pts.ndim != points.ndim:
                    add_pts = np.expand_dims(add_pts, 0)
                full_pts = np.concatenate((points, add_pts))
                ones = np.zeros(len(points), dtype=int) + 1
                if len(np.shape(add)) == 0:
                    zeros = np.zeros(1, dtype=int)
                else:
                    zeros = np.zeros(len(add), dtype=int)
                hole_array = np.concatenate((ones, zeros))
                return full_pts, hole_array
            else:
                add = np.array(pot_solutions[sorts[0]][0])
                add_pts = pot_holes[add]
                if add_pts.ndim == 1:
                    add_pts = np.expand_dims(add_pts, 0)
                temp_full_pts = np.concatenate((points, add_pts))
                connected_indexes = get_connected_indexes(temp_full_pts)

        else:
            add_pts = []

        if 'temp_full_pts' not in locals():
            temp_full_pts = points

        combs = itertools.combinations(np.arange(len(connected_indexes)), 2)
        for comb in combs:
            lccs = []
            # print(points, connected_indexes, comb)
            n_A = temp_full_pts[connected_indexes[comb[0]]]
            n_B = temp_full_pts[connected_indexes[comb[1]]]
            trials = cartesian_product(np.arange(len(n_A)), np.arange(len(n_B)))
            for t in trials:
                lcc = least_common_containee(n_A[t[0]], n_B[t[1]])
                lccs.append(lcc)
            min_ind = np.argmin([np.sum(i) for i in lccs])
            A = n_A[trials[min_ind][0]]
            B = n_B[trials[min_ind][1]]
            lcc = lccs[min_ind]
            path_from_A = possible_paths(A, lcc)[0]
            path_from_B = possible_paths(B, lcc)[0]
            adds = np.concatenate((np.expand_dims(lcc, 0), path_from_A, path_from_B))
            if len(add_pts) == 0:
                add_pts = adds
            else:
                add_pts = np.concatenate((add_pts, adds))
        add_pts = npi.unique(add_pts)
        full_pts = np.concatenate((temp_full_pts, add_pts))
        ones = np.zeros(len(points), dtype=int) + 1
        zeros = np.zeros(len(full_pts) - len(points), dtype=int)
        hole_array = np.concatenate((ones, zeros))
        return full_pts, hole_array
        # currently doesn't work when there are three seperate groups, in second category,
        # and you only have to connect a to b and b to c; not also  c to a


def get_layers(points):
    """For a 3d collection of tones, splits the points into layers. Returns a
    list of lists of coordinates, one list for each layer."""

    # first split into layers
    sums = np.sum(points, axis=1)
    max_layer = np.max(sums)
    layers = [points[sums==layer] for layer in range(max_layer+1)]
    return layers

def get_layer_skew(points):
    """Given a 3d collection of tones, return a array of arrays, one for each
    layer, that describes the relative skew of that layer toward each dimension."""
    layers = get_layers(points)
    skew=[]
    for i, layer in enumerate(layers):
        sk = np.sum(layer, axis=0)
        if np.sum(sk) != 0:
            sk = sk / np.sum(sk)
        skew.append(sk)
    return(np.array(skew))


def root_salience(points, scaled=True, indexes=False):
    """Given a chord / neighborhood, return the roots, the number of tones
    that each root contains, and (if indexes==True), the indexes of each root.
    By default, the 'weight' of each root is scaled by dividing by the total
    number of non-root points."""

    roots = points[are_roots(points)]
    non_roots = points[np.invert(are_roots(points))]
    weight = []
    for r in roots:
        ct = 0
        for nr in non_roots:
            if is_contained_by(nr, r):
                ct += 1
        weight.append(ct)
    if scaled:
        weight = [i/len(non_roots) for i in weight]
    if indexes:
        return roots, weight, np.nonzero(are_roots(points))[0]
    else:
        return roots, weight


def dc_alg_step(size, counts=None, alpha=1.0):
    """Performs a single iteration of James Tenney's Dissonant Counterpoint
    Algorithm. Randomly chooses an index from range(size) with weights based on
    the previous 'counts'. The process starts off with counts all = 1. Each time
    an element is chosen, its count goes down to 1. Each time an element is not
    chosen, its count is incremented up by one.

    Parameters:
        size (integer): the number of elements to choose between.
        counts (array of ints >= 1): the counts of each element.
        alpha (float): the 'sharpness' of the weighting.

    """
    if np.all(counts == None):
        counts = np.zeros(size, dtype=int) + 1

    weight = counts**alpha
    p = weight / np.sum(weight)
    choice_index = np.random.choice(np.arange(size), p=p)
    counts += 1
    counts[choice_index] = 1
    return choice_index, counts


def dc_alg(size, epochs, counts=None, alpha=1.0, return_counts=False):
    """Iterates through multiple dc_alg_steps, returning the list of element
    indexes, and (if requested) the number of counts at the end.

    Parameters:
        size (integer): the number of elements to choose between.
        counts (array of ints >= 1): the counts of each element.
        alpha (float > 0.0): the 'sharpness' of the weighting.
        epochs (integer): the number of times to iterate through the dc_alg."""

    choices = []
    for e in range(epochs):
        choice, counts = dc_alg_step(size, counts, alpha)
        choices.append(choice)
    choices = np.array(choices)
    if return_counts:
        return choices, counts
    else:
        return choices


def group_dc_alg_step(groups, group_counts=None, element_counts=None, alpha=1.0):
    """dc_alg, but for groups of elements.

    Parameters:
        groups (array of nested arrays, each filled with ints)
    """
    if np.all(group_counts == None):
        group_counts = np.zeros(len(groups), dtype=int) + 1
    if np.all(element_counts == None):
        element_counts = np.zeros(np.max(flatten(groups))+1, dtype=int) + 1

    # avg of element counts for each group
    group_avg_ec = []
    for group in groups:
        avg = np.mean([element_counts[i] for i in group])
        group_avg_ec.append(avg)
    group_avg_ec = np.array(group_avg_ec)
    weight = group_counts * group_avg_ec
    p = weight/ np.sum(weight)
    choice_index = np.random.choice(np.array(len(groups)), p=p)
    choice = groups[choice_index]
    group_counts += 1
    group_counts[choice_index] = 1
    element_counts += 1
    for i in choice:
        element_counts[i] = 1
    return choice_index, group_counts, element_counts

def group_dc_alg(groups, epochs, group_counts=None, element_counts=None,
                 alpha=1.0, return_counts=False):
    cis = []
    gc = group_counts
    ec = element_counts
    for e in range(epochs):
        choice_index, gc, ec = group_dc_alg_step(groups, gc, ec, alpha)
        cis.append(choice_index)
    if return_counts:
        return [groups[i] for i in cis], gc, ec
    else:
        return [groups[i] for i in cis]


# def group_member_dc_alg_step(groups, )

def make_random_trajectory(length, max_step=1, dims=3, circular=True):
    """

    parameters:
        length (integer, if circular==True, must be even)

    """
    steps = np.zeros((length, dims), dtype=int)
    indexes = np.random.randint(dims, size=length)
    steps[np.arange(length), indexes] = np.random.randint(-1, 2, size=length)
    if circular == True:
        mirror = -1 * steps[:int(length/2)]
        np.random.shuffle(mirror)
        steps[int(length/2):] = mirror
    return steps


def traj_to_absolute(traj):
    origin = np.zeros(shape=(1, np.shape(traj)[-1]), dtype=int)
    return np.concatenate(((origin), np.cumsum(traj, axis=0)))


def hsv_to_freq(hsv, primes, fund, oct=(0, 0, 0)):
    oct = np.array(oct)
    if len(np.shape(hsv)) == 2:
        # print(hsv)
        sub_prod = (primes ** hsv) * (2.0 ** (hsv * oct))
        freq = fund * np.prod(sub_prod, axis=1)
    elif len(np.shape(hsv)) == 3:
        sub_prod = (primes ** hsv) * (2.0 ** (hsv * oct))
        freq = fund * np.prod(sub_prod, axis=2)

    else:
        freq = fund * np.prod((primes ** hsv) * (2.0 ** (hsv * oct)))
    return freq

def hsv_to_gen_ratios(hsv, primes):
    """For a list of tones specified as harmonic series vectors, return the
    associated octave-generalized ratios."""
    out = np.prod(primes ** hsv, axis=1)
    while np.any(out < 1) or np.any(out >= 2):
        out = np.where(out < 1, out * 2, out)
        out = np.where(out >=2, out / 2, out)
    return out



def octave_finder(chord, fund, primes, lims=(50, 50*(2**5)), max_width=False):
    """Returns all of the possible octave shifts that would make the notes in
    the chord audible."""
    bit = np.arange(-4, 4)
    cp = cartesian_product(*(bit for i in range(np.shape(chord)[-1])))
    seive = np.zeros(len(cp), dtype=bool)
    freq_ranges=[]
    for i, oct in enumerate(cp):
        freq = hsv_to_freq(chord, primes, fund, oct)
        min_freq = np.min(freq)
        max_freq = np.max(freq)
        freq_ranges.append(max_freq - min_freq)
        if np.all(min_freq >= lims[0]) and np.all(max_freq <= lims[1]):
            seive[i] = True
    possible_octs = cp[seive]
    freq_ranges = np.array(freq_ranges)
    freq_ranges = freq_ranges[seive]
    if max_width:
        return possible_octs, freq_ranges
    else:
        return possible_octs

# timbre tools

def fill_in_octaves(freqs, max_freq=None, as_harms=False):
    """For an array of frequencies, return an array will the original array plus
    all octave multiples of each that are less than the maximum frequency."""
    if max_freq == None:
        max_freq = np.max(freqs)
    out = []
    for freq in freqs:
        ct = 0
        while freq * (2 ** ct) <= max_freq:
            out.append(freq * (2 ** ct))
            ct += 1
    out = np.array(out)
    if as_harms:
        partials = np.arange(1, max_freq + 1)
        truth = npi.contains(out, partials)
        return truth.astype(int)
    else:
        return out
