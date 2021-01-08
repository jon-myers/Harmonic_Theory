from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib.colors
import matplotlib.tri as mtri
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.spatial.distance import cdist
import numpy as np
import itertools, ternary
import random
from fractions import Fraction
from utils import traj_to_point_tuples, traj_to_points, get_segments, \
                  get_ratios, is_contained_by, are_roots, are_extremities, \
                  get_layers

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def make_plot(pts, path, primes=[2, 3, 7], octaves = None, draw_points = None,
              oct_generalized = False, dot_size=1, colors=None, ratios=True,
              origin=False, origin_range = [-2, 3], get_ax=False, legend=True,
              range_override=[0, 0], transparent=True, connect_color='grey',
              draw_point_visible=False, draw_color='seagreen', connect_size=1,
              file_type='pdf', opacity=0.5, root_layout=False, elev=16,
              azim=-72, type='none', labels=False, label_offset=[-0.25, 0.2, 0],
              layer_slices=[], layer_color='grey'):


    c = matplotlib.colors.get_named_colors_mapping()
    if np.all(colors == None):
        colors = ['black' for i in range(len(pts))]
    else:
        colors = [c[i.lower()] for i in colors]

    if type != 'none':
        colors = get_colors(pts, type)
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

    for slice in layer_slices:
        add_slice_to_ax(slice, ax, layer_color=layer_color)

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
        if labels == True:

            l_pt = pt + np.array(label_offset)
            ax.text(l_pt[0], l_pt[1], l_pt[2], str(i+1), fontsize=12)
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

def make_2d_plot(points, path, axis_range=[-1, 3], origin=True):
    fig = plt.figure(figsize=[6, 6])
    ax = fig.add_subplot(111)
    if origin:
        ax.arrow(axis_range[0], 0, axis_range[1] - axis_range[0], 0, color='black',
                 length_includes_head=True, width=0.001, head_width=0.1)
        ax.arrow(0, axis_range[0], 0, axis_range[1] - axis_range[0], color='black',
                 length_includes_head=True, width=0.001, head_width=0.1)
        ticks = [i for i in range(axis_range[0]+1, axis_range[1])]
        ax.scatter(ticks, [0 for i in range(len(ticks))], color='black',
                   marker='|', s=40)
        ax.scatter([0 for i in range(len(ticks))], ticks, color='black',
                   marker='_', s=40)

    segs = get_segments(points)
    for seg in segs:
        ax.plot(seg[0], seg[1], color='black', alpha=1, lw=3)
    ax.scatter(points[:, 0], points[:, 1], color='black', s=150)


    ax.set_axis_off()
    ax.set_xlim(*axis_range)
    ax.set_ylim(*axis_range)
    plt.savefig(path + '.pdf', transparent=True)

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


def plot_tree(points, path, type='root', labels=False):
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
    colors = get_colors(points, type)
    colors = [colors[i] for i in edge_order]

    pos=graphviz_layout(G, prog='dot')
    plt.figure(figsize=[3, 4])
    nx.draw(G, pos, arrows=False, node_color=colors)
    if labels == True:
        numbers = [i+1 for i in range(len(points))]
        numbers = [str(numbers[i]) for i in edge_order]
        x_vals, y_vals = zip(*pos.values())
        x_max = max(x_vals)
        x_min = min(x_vals)
        y_max = max(y_vals)
        y_min = min(y_vals)
        x_margin = 0.15 * (x_max - x_min)
        y_max = max(y_vals)
        y_min = min(y_vals)
        y_margin = 0.1 * (y_max - y_min)
        offset_pos = {}
        label_pos = {}
        for key in pos.keys():
            offset_pos[key] = (pos[key][0] - 25, pos[key][1] + 5)
            label_pos[key] = key+1

        nx.draw_networkx_labels(G, pos=offset_pos, labels=label_pos)
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.savefig(path + '.pdf', transparent=True)
    plt.close()

def get_colors(points, type='root'):
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
    return colors


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

def plot_simple_trajectory(traj, path, root=None, range_override=[-1, 3],
    file_type='pdf', transparent=True, arrow_proportion=0.7, dot_color='black',
    arrow_color='blue'):
    if np.all(root == None):
        root = np.zeros(np.shape(traj)[-1], dtype=int)

    fig = plt.figure(figsize=[8, 6])
    ax = mplot3d.Axes3D(fig, elev=16, azim=-72)

    points = traj_to_points(traj)
    steps = traj_to_point_tuples(traj)

    for pt in points:
        ax.scatter(pt[0], pt[1], pt[2], color='black', depthshade=False,
        s=int(60 * dot_size))


    for step in steps:
        np_step = np.array(step)
        changing_index = np.nonzero(np_step[0] != np_step[1])
        start = np_step[0, changing_index]
        end = np_step[1, changing_index]
        delta = end - start
        alt_start = start + (1 - arrow_proportion) * delta / 2
        alt_end = end - (1 - arrow_proportion) * delta / 2
        np_step[:, changing_index] = [alt_start, alt_end]
        arrow = Arrow3D(*np_step.T, mutation_scale=10, lw=1, arrowstyle='-|>',
                        color=arrow_color)
        ax.add_artist(arrow)

    points = traj_to_points(traj)
    for pt in points:
        ax.scatter(*pt, color=dot_color, depthshade=False, s=int(60))
    # x_arrow = Arrow3D([0, 1], [0, 0], [0, 0], mutation_scale=20, lw=1, arrowstyle='-|>', color='black')
    # ax.add_artist(x_arrow)

    ax.set_axis_off()

    ax.set_xlim3d(range_override)
    ax.set_ylim3d(range_override)
    ax.set_zlim3d(range_override)


    plt.savefig(path + '.' + file_type, transparent=transparent)
    plt.close()


def make_4d_plot(points, path):
    """Makes a series of plots for each 'slice' of a 4 dimensional harmonic
    space lattice. That is, makes a 3d plot for all of the points in which the
    4th dimension is 0; a 3d plot for all of the poitns in which the 4th
    dimension is 1; etc. Plots will be presented next to each other horizontally,
    with adjacent slice plot points transparent."""
    # how many of the 4th axis?
    slice_vals = np.unique(points[:, -1])
    slice_vals = np.arange(np.min(slice_vals), np.max(slice_vals) + 1)
    slices = [points[points[:, -1] == i][:, :-1] for i in slice_vals]
    figsize = [3 * len(slice_vals), 3]
    fig = plt.figure(figsize=figsize)
    all_segments = [get_segments(slice) for slice in slices]
    axis_range = [np.min(points), np.max(points)]

    for i, slice in enumerate(slice_vals):
        ax = fig.add_subplot(1, len(slice_vals), i+1, projection='3d')
        ax.view_init(elev=16, azim=-72)
        other_indexes = [ind for ind in range(len(slice_vals)) if ind != i]

        for oi in other_indexes:
            for seg in all_segments[oi]:
                ax.plot(seg[0], seg[1], seg[2], color='lightgrey', alpha=0.5, lw=2)
            for pt in slices[oi]:
                ax.scatter(*pt, color='lightgrey', depthshade=False, s=int(30))

        for seg in all_segments[i]:
            ax.plot(seg[0], seg[1], seg[2], color='black', alpha=0.5, lw=2)
        for pt in slices[i]:
            ax.scatter(*pt, color='black', depthshade=False, s=int(30))

        ax.set_axis_off()
        ax.set_xlim3d(axis_range)
        ax.set_ylim3d(axis_range)
        ax.set_zlim3d(axis_range)

    plt.savefig(path + '.pdf', transparent=True)
    plt.close()




test_points = np.array((
    (0, 0, 0, 0),
    (0, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, -1),
    (2, 0, 0, -1),
    (0, -1, 1, 0)
))

points, hole_arr = fix_collection(test_points)

permutations = np.array([i for i in itertools.permutations(range(4))])
for i, perm in enumerate(permutations):
    points = points[:, perm]

    make_4d_plot(points, 'perms/'+str(i))

# test_pts = np.array((
#     (0, 0, 0),
#     (0, 0, 1),
#     (3, 0, 0),
#     (2, -1, 0),
#     (1, 0, 0),
#     (0, 1, 0),
#     (2, 0, 0)))
# # o = create_tree_edges(test_pts)
# # print(o)
# plot_tree(test_pts, 'test')

def add_slice_to_ax(layer, ax, layer_color='grey'):
    xy=np.array(((0, 0), (layer, 0), (0, layer)))
    triang = mtri.Triangulation(xy[:,0], xy[:, 1])
    z = [layer, 0, 0]
    ax.plot_trisurf(triang, z, color=layer_color, alpha=0.5)
    ax.plot([0, layer, 0, 0], [0, 0, layer, 0], [layer, 0, 0, layer], color='black')


def plot_ternary(points, path):
    """Plots the ternary slices of a collection of points. (Collection must
    be 'cast to ordinal', or at least non-negative, in order to work)."""
    layers = get_layers(points)
    fig, axes = plt.subplots(1, len(layers),figsize=[3*len(layers), 3])
    if len(layers[0]) == 1:
        axes[0].scatter(0, 0, color='black')
    axes[0].xaxis.set_visible(False)
    axes[0].yaxis.set_visible(False)
    axes[0].set_title('Layer 0')
    for i, layer in enumerate(layers[1:]):

        figure, tax = ternary.figure(scale=i+1, ax=axes[i+1])
        lines = np.linspace(0, i+1, 500)
        for line in lines:
            tax.horizontal_line(line, color='grey', alpha=0.05)
        tax.boundary(linewidth=1.0)
        tax.gridlines(color='grey', multiple=1)
        tax.clear_matplotlib_ticks()


        tax.scatter(layer, color='black')
        tax.set_title('Layer ' + str(i+1))
    plt.savefig(path + '.pdf', transparent=True)


# plot_ternary(points, 'ternary_test')

# scale=2
# figure, tax = ternary.figure(scale=scale)
# tax.boundary(linewidth=2.0)
# tax.gridlines(color='grey', multiple=1)
#
# # tax.ticks(axis='lbr', linewidth=1)
# tax.clear_matplotlib_ticks()
# points = [(1, 1, 0)]
# tax.scatter(points)
#
# ternary.plt.show()
