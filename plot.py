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
import numpy as np
from utils import traj_to_point_tuples, traj_to_points

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

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

def plot_simple_trajectory(traj, path, root=None, range_override=[-1, 3],
<<<<<<< Updated upstream
    file_type='pdf', transparent=True, arrow_proportion=0.7, dot_color='black',
    arrow_color='blue'):
=======
    file_type='pdf', transparent=True, dot_size=1):
>>>>>>> Stashed changes
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

# test_traj = np.array(((1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, -1)))
#
# plot_simple_trajectory(test_traj, 'test')
