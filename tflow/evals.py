import matplotlib.pyplot as plt
import os
import numpy as np


def shiftflip(points, cls):
    if cls == 0:
        # flip
        points[..., 1] = 1 - points[..., 1]
        # shift
        points[..., :] = points[..., :] + [-1., -0.5]
    else:
        # shift
        points[..., :] = points[..., :] - [-1., -0.5]
        # flip
        points[..., 1] = 1 - points[..., 1]
    return points


def shift_flip_sickness(root, din, dout, config):
    '''Test how the shift and flip of the classes works out when inverting a
    class.'''

    X_orig = X = din.labels['X']
    cls = y = din.labels['y']

    X_switch = dout.labels['X_9_switch']

    choice = np.random.choice(len(X), size=200, replace=False)

    X_orig = X = X[choice]
    cls = y = cls[choice]
    X_switch = X_switch[choice]

    f, AX = plt.subplots(2, 2,
                         figsize=[12.8, 7.2], dpi=300, constrained_layout=True)

    [[ax1, ax2], [ax3, ax4]] = AX

    color = c = X[..., 0] * 2 + X[..., 1]

    ax1.scatter(X_orig[..., 0], X_orig[..., 1], alpha=0.1, label='all observations', c=c)
    ax2.scatter(X_switch[..., 0], X_switch[..., 1], alpha=0.1, label='switched classes', c=c)

    X_0 = X_orig[y == 0]
    X_1 = X_orig[y == 1]
    X_1_sf = shiftflip(X_orig[y == 1], 1)
    c0 = color[y == 0]
    c1 = color[y == 1]

    X_s0 = X_switch[y == 0]
    X_s1 = X_switch[y == 1]

    X_s0_sf = shiftflip(X_switch[y == 0], 0)
    X_s1_sf = shiftflip(X_switch[y == 1], 1)

    ax3.scatter(X_0[..., 0], X_0[..., 1], alpha=0.1, label='X cls 0', c=c0, marker='o', ec='k')
    ax3.scatter(X_s0[..., 0], X_s0[..., 1], alpha=0.1, label='X switch cls 0', c=c0, marker='s', ec='k')
    ax3.scatter(X_s0_sf[..., 0], X_s0_sf[..., 1], alpha=0.1, label='shiftflip(X switch cls 0)', c=c0, marker='s', ec='k')

    ax4.scatter(X_1[..., 0], X_1[..., 1], alpha=0.1, label='X cls 1', c=c1, marker='o', ec='k')
    ax4.scatter(X_s1[..., 0], X_s1[..., 1], alpha=0.1, label='X switch cls 1', c=c1, marker='s', ec='k')
    ax4.scatter(X_s1_sf[..., 0], X_s1_sf[..., 1], alpha=0.1, label='shiftflip(X switch cls 1)', c=c1, marker='s', ec='k')


    obs_idxs = np.random.choice(np.arange(len(X_0)), size=10, replace=False)
    obs = X_0[obs_idxs]
    sort_idxs = np.argsort(obs[..., 0])
    obs = obs[sort_idxs]

    NNs = X_s0[obs_idxs][sort_idxs]
    NNs_sf = X_s0_sf[obs_idxs][sort_idxs]

    ax3.scatter(obs[..., 0], obs[..., 1], alpha=1.0, label='observation', c=np.arange(10), marker='o', ec='k')
    ax3.scatter(NNs[..., 0], NNs[..., 1], alpha=1.0, label='NN sequence', c=np.arange(10), marker='s', ec='k')
    ax3.scatter(NNs_sf[..., 0], NNs_sf[..., 1], alpha=0.5, label='shift flipped NN sequence', c=np.arange(10), marker='^', ec='k')

    dist_X_1_X_s1_sf = np.mean(0.5*np.sqrt(((X_1 - X_s1_sf) ** 2).sum(-1)))
    ax4.set_title(f'mean dist \noriginal vs shift-flipped class inverted points: \n{dist_X_1_X_s1_sf:0.3f}')

    dist_X_0_X_s0_sf = np.mean(0.5*np.sqrt(((X_0 - X_s0_sf) ** 2).sum(-1)))
    ax3.set_title(f'mean dist \n original vs shift-flipped class inverted points:\n {dist_X_0_X_s0_sf:0.3f}')

    for ax in AX.flatten():
        ax.legend()
        ax.set_aspect(1)

    f.savefig(os.path.join(root, 'shift_flip_test_on_moon.png'))
