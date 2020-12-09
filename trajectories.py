import numpy as np
import numpy_indexed as npi
from utils import cast_traj_to_ordinal, ult_vector, get_directionality, \
    get_crossings, traj_to_points

def make_simple_trajectories(epochs, dims):
    """Generates all possible unique simple trajectories for a given number of
    steps in a harmonic space lattice with a given number of dimensions. By
    simple I mean trajectories that only incude steps of one unit, in one
    dimension at a time."""
    layers = np.zeros((2, 1, dims), dtype=int)
    layers[0, 0, 0] = 1
    layers[1, 0, 0] = -1
    layers = [layers]
    eye = np.eye(dims, dtype=int)
    steps = np.concatenate((eye, -1 * eye))

    for e in range(epochs-1):
        last_trajes = layers[-1]
        t_steps = np.broadcast_to(steps, (np.shape(last_trajes)[0], *np.shape(steps)))
        t_steps = np.expand_dims(t_steps, -2)
        lt = np.expand_dims(last_trajes, 1)
        t_sh = np.shape(t_steps)
        lt_sh = np.shape(lt)
        lt = np.broadcast_to(lt, (t_sh[0], t_sh[1], lt_sh[2], t_sh[3]))
        next_steps = np.concatenate((lt, t_steps), axis=2)
        sh = np.shape(next_steps)
        next_steps = np.reshape(next_steps, (sh[0] * sh[1], sh[2], sh[3]))
        next_steps = np.array([cast_traj_to_ordinal(i) for i in next_steps])
        next_steps = npi.unique(next_steps)
        layers.append(next_steps)
    return layers

# trajes = make_simple_trajectories(6, 3)
# # for t in trajes[-1]:
# #     print(t)
# #     print(ult_vector(t))
# #     print(get_directionality(t))
#
# a = trajes[-1][6]
# for a in trajes[-1]:
#     dir = get_directionality(a)
#     get_crossings(a)
#     pts, persistence = traj_to_points(a, unique=False, persistence=True)
#     print(pts, persistence)
