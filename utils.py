import itertools
from fractions import Fraction
import numpy as np
import numpy_indexed as npi
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import math
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

def make_plot(pts, primes, path, octaves = None, draw_points = None,
              oct_generalized = False, dot_size=1, colors=None, ratios=True,
              origin=False, origin_range = [-2, 3], get_ax=False, legend=True,
              range_override=[0, 0], transparent=False, connect_color='grey',
              draw_point_visible=False, draw_color='seagreen', connect_size=1,
              file_type='pdf', opacity=0.5, root_layout=False):


    c = matplotlib.colors.get_named_colors_mapping()
    if np.all(colors == None):
        colors = ['black' for i in range(len(pts))]
    else:
        colors = [c[i.lower()] for i in colors]
    if np.all(octaves == None):
        octaves = np.repeat(0, len(primes))
    if np.all(draw_points == None):
        # if root_layout == True and len(rl_draw_points) > 0:
        #     segments = get_segments(np.concatenate((pts, rl_draw_points)))
        # else:
        segments = get_segments(pts)
    else:
        # if root_layout == True and len(rl_draw_points) > 0:
        #     segments = get_segments(np.concatenate((pts, draw_points, rl_draw_points)))
        # else:
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

def get_transpositions(points):
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

def cast_to_ordinal(points):
    # needs to be fixed, not discriminating enough!
    """Given a list of harmonic space vectors, returns the collection cast to
    ordinal position. That is, with dimensions sorted firstly by max extent from
    origin, and secondly by average extent in each dimension, and thirdly by
    the order of extent in each dimension of point with furthers manhattan
    distance"""

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
    return points

def reorder_points(points):
    """Reorders points such that they are in a consistent order for testing for
    uniqueness against other sets of points."""
    primes = np.array((2, 3, 5, 7, 11, 13, 17, 19))[:np.shape(points)[-1]]
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

    # find an octave shift vector such that sum(osv * hsvs) = octs

primes, hsvs, shifts = analyze([33/2, 7, 22, 21/2])

print(primes, '\n\n', hsvs, '\n\n', shifts)
