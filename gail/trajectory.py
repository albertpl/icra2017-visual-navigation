import numpy as np


class Trajectory(object):
    __slots__ = ('obs', 'a_dists', 'actions', 'rewards')

    def __init__(self, obs, a_dists, actions, rewards):
        assert (
            obs.ndim == 3 and a_dists.ndim == 2 and actions.ndim == 1 and rewards.ndim == 1 and
            obs.shape[0] == a_dists.shape[0] == actions.shape[0] == rewards.shape[0]
        )
        self.obs = obs
        self.a_dists = a_dists
        self.actions = actions
        self.rewards = rewards

    def __len__(self):
        return self.obs.shape[0]


def raggedstack(arrays, fill=0., axis=0, raggedaxis=1):
    """
    Stacks a list of jagged arrays, like np.stack with axis=0.
    Arrays may have different length (along the raggedaxis), and will be padded on the right
    with the given fill value.
    """
    assert axis == 0 and raggedaxis == 1, 'not implemented'
    arrays = [a[None, ...] for a in arrays]
    assert all(a.ndim >= 2 for a in arrays)

    out_shape = list(arrays[0].shape)
    out_shape[0] = sum(a.shape[0] for a in arrays)
    out_shape[1] = max(a.shape[1] for a in arrays)  # take max along ragged axes
    out_shape = tuple(out_shape)

    out = np.full(out_shape, fill, dtype=arrays[0].dtype)
    pos = 0
    for a in arrays:
        out[pos:pos+a.shape[0], :a.shape[1], ...] = a
        pos += a.shape[0]
    assert pos == out.shape[0]
    return out


class RaggedArray(object):
    def __init__(self, arrays, lengths=None):
        if lengths is None:
            # Without provided lengths, `arrays` is interpreted as a list of arrays
            # and self.lengths is set to the list of lengths for those arrays
            self.arrays = arrays
            self.stacked = np.concatenate(arrays, axis=0)
            self.lengths = np.array([len(a) for a in arrays])
        else:
            # With provided lengths, `arrays` is interpreted as concatenated data
            # and self.lengths is set to the provided lengths.
            self.arrays = np.split(arrays, np.cumsum(lengths)[:-1])
            self.stacked = arrays
            self.lengths = np.asarray(lengths, dtype=int)
        assert all(len(a) == l for a, l in zip(self.arrays, self.lengths))
        self.boundaries = np.concatenate([[0], np.cumsum(self.lengths)])
        assert self.boundaries[-1] == len(self.stacked)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.stacked[self.boundaries[idx]:self.boundaries[idx+1], ...]

    def padded(self, fill=0.):
        return raggedstack(self.arrays, fill=fill, axis=0, raggedaxis=1)


class TrajBatch(object):
    def __init__(self, trajs, obs, adist, a, r, time):
        self.trajs, self.obs, self.adist, self.a, self.r, self.time = \
            trajs, obs, adist, a, r, time

    @classmethod
    def from_trajs(cls, trajs):
        assert all(isinstance(traj, Trajectory) for traj in trajs)
        obs = RaggedArray([t.obs for t in trajs])
        adist = RaggedArray([t.a_dists for t in trajs])
        a = RaggedArray([t.actions for t in trajs])
        r = RaggedArray([t.rewards for t in trajs])
        time = RaggedArray([np.arange(len(t), dtype=float) for t in trajs])
        return cls(trajs, obs, adist, a, r, time)

    def with_replaced_reward(self, new_r):
        new_trajs = [Trajectory(traj.obs, traj.a_dists, traj.actions, traj_new_r)
                     for traj, traj_new_r in zip(self.trajs, new_r)]
        return TrajBatch(new_trajs, self.obs, self.adist, self.a, new_r, self.time)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]
