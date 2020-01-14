from edflow.util import edprint, retrieve
from edflow.custom_logging import get_logger

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

from tflow.util import Seq2Vec


logger = get_logger(__name__)


def plot_sequences(root, di, do, config):
    prng = np.random.RandomState(42)
    n_seq = retrieve(config, 'eval/seq/n', default=10)
    n_t = retrieve(config, 'model_pars/n_transformers')

    s2v = Seq2Vec(100, 2)

    choice = prng.choice(len(di), size=n_seq, replace=False)

    for tid in tqdm(choice, desc='T'):

        X_in = di.labels['X'][tid]
        logger.info(f'X {X_in.shape}')
        cls = di.labels['y'][tid]
        logger.info(f'X_in: {X_in.shape}')
        logger.info(f'cls: {cls.shape}')
        cls_str = f'health: {list(set(cls))[0]}'
        inv_cls_str = f'health: {1 - list(set(cls))[0]}'

        Y_in = do.labels['Y'][tid]
        logger.info(f'Y {Y_in.shape}')
        Y_in_s = s2v.inverse(Y_in)
        logger.info(f'Y {Y_in_s.shape}')

        X_out_switch = do.labels[f'X_{n_t-1}_switch'][tid]
        logger.info(f'Xo {X_out_switch.shape}')

        f, [[ax1, ax2], [ax3, ax4]] = _, AX = plt.subplots(
            2, 2,
            figsize=[12.8, 7.2], dpi=300,
            constrained_layout=True
        )

        ax1.scatter(X_in[..., 0], X_in[..., 1])
        ax1.set_title(f'Input sequence ({cls_str})')

        ax2.scatter(X_out_switch[..., 0], X_out_switch[..., 1])
        ax2.set_title(f'Output sequence ({inv_cls_str})')

        ax3.scatter(Y_in_s[..., 0], Y_in_s[..., 1])
        ax3.set_title(f'Gaussianised Sequence ({cls_str})')

        ax4.axis('off')

        for ax in [ax1, ax2]:
            ax.set_ylim(0, 1.1)
            ax.set_xlim(0, 1.1)
            ax.set_aspect(1)

        savepath = os.path.join(root, f'traj_{tid}.png')
        f.savefig(savepath)
        logger.info(f'Saved plot to {savepath}.')
