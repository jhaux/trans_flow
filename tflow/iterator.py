from edflow.iterators.template_iterator import TemplateIterator
from edflow.util import edprint, retrieve
import torch
import os
import numpy as np

import matplotlib.pyplot as plt

from tflow.util import t2np, np2t, LazyT2OH


t2oh = LazyT2OH()


class Iterator(TemplateIterator):
    def __init__(self, config, root, model, dataset, **kwargs):
        self.model = model
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)  #, momentum=0.9)
        
        super().__init__(config, root, model, dataset, **kwargs)
        
    def step_op(self, _, labels_, **kwargs):
        X = np2t(labels_['X'])
        cls = np2t(labels_['y'],
                   cuda=False,
                   to_float=False).view(X.shape[0], 1).long()
        cls = t2oh(cls, 2).cuda()
        
        Y = self.model(X, cls) if self.model.is_cond else self.model(X)

        log_det_loss = -torch.mean(self.model.log_det)
        
        kl_loss = torch.mean(0.5 * torch.sum(Y ** 2, dim=-1))
        
        loss = kl_loss + log_det_loss

        def train_op():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        def eval_op():
            Y_intermediate = [t2np(i) for i in self.model.intermediates[:-1]]

            Y_sample_np = np.random.normal(size=[64, 2])
            Y_sample = torch.from_numpy(Y_sample_np).float().cuda()

            X_inv = self.model.inv(Y_sample) if not self.model.is_cond else self.model.inv(Y_sample, cls)
            X_intermediate = [t2np(i) for i in self.model.intermediates[:-1]]

            ret_d = {'labels': {
                'Y': t2np(Y),
                'Y_sample': Y_sample_np,
                'X_sample': t2np(X_inv),
                'pos_sample': t2np(X_inv)
            }}

            for i, Y_i in enumerate(Y_intermediate):
                ret_d['labels'][f'Y_{i}'] = Y_i

            for i, X_i in enumerate(X_intermediate):
                ret_d['labels'][f'X_{i}'] = X_i

            if self.model.is_cond:
                cls_0 = t2oh(torch.zeros_like(cls.cpu()).long(), 2).cuda()
                cls_1 = t2oh(torch.ones_like(cls.cpu()).long(), 2).cuda()

                # Case 1: only class 0
                X_inv = self.model.inv(Y_sample, cls_0)
                X_interm = [t2np(i) for i in self.model.intermediates]

                for i, X_i in enumerate(X_interm):
                    ret_d['labels'][f'X_{i}_0-0'] = X_i

                # Case 2: only class 1
                X_inv = self.model.inv(Y_sample, cls_1)
                X_interm = [t2np(i) for i in self.model.intermediates]

                for i, X_i in enumerate(X_interm):
                    ret_d['labels'][f'X_{i}_1-1'] = X_i

                # Case 3: invert class 
                X_inv = self.model.inv(Y, 1 - cls)
                X_interm = [t2np(i) for i in self.model.intermediates]

                for i, X_i in enumerate(X_interm):
                    ret_d['labels'][f'X_{i}_switch'] = X_i

            return ret_d
        
        def log_op():
            return {'scalars':
                    {'loss': loss, 'kl': kl_loss, 'det': log_det_loss}
                    }
        
        return {'train_op': train_op, 'eval_op': eval_op, 'log_op': log_op}
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)

        self.model.cuda()


def mscatter(x,y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def plot_callback(root, data_in, data_out, config):
    n_t = retrieve(config, 'model_pars/n_transformers')

    label = data_in.labels['y']

    X_intermediate = []
    Y_intermediate = []
    for i in range(n_t-1):
        X_intermediate += [data_out.labels[f'X_{i}']]
        Y_intermediate += [data_out.labels[f'Y_{i}']]

    X_in = data_in.labels['X']
    Y_out = data_out.labels['Y']
    Y_sample = data_out.labels['Y_sample']
    X_sample = data_out.labels['X_sample']

    X2Y = [X_in] + Y_intermediate + [Y_out]
    Y2X = [Y_sample] + X_intermediate + [X_sample]

    prng = np.random.RandomState(42)
    choice = prng.choice(len(X_in), size=200, replace=False)

    X2Y = [t[choice] for t in X2Y]
    Y2X = [t[choice] for t in Y2X]

    label = label[choice]

    color_f = X2Y[0][..., 0] * (X2Y[0][..., 1].max() - X2Y[0][..., 1].min()) + X2Y[0][..., 1]
    color_b = Y2X[0][..., 0] * (Y2X[0][..., 1].max() - Y2X[0][..., 1].min()) + Y2X[0][..., 1]

    ms = ['s', 'o']
    marker_f = [ms[c] for c in data_in.labels['y'][choice]]
    marker_b = 'o'

    w = 5
    f, AX = plt.subplots(len(X2Y), 2,
                         sharex=True, sharey=True,
                         figsize=(w, 0.6*len(X2Y) * w),
                         constrained_layout=True)

    for i, Ax in enumerate(AX):
        for j, ax in enumerate(Ax):
            points = X2Y[i] if j == 0 else Y2X[n_t - i]
            color = color_f if j == 0 else color_b

            marker = marker_f if j == 0 else marker_b

            mscatter(points[..., 0], points[..., 1], ax, marker, c=color,
                     ec='k')
            ax.set_aspect(1)
            ax.set_title(i if j == 0 else n_t - i)

    f.savefig(os.path.join(root, 'in_n_out.png'), dpi=300)


def plot_callback_cond(root, data_in, data_out, config):
    n_t = retrieve(config, 'model_pars/n_transformers')

    label = data_in.labels['y']

    X_intermediate = []
    Y_intermediate = []
    X_cls0 = []
    X_cls1 = []
    X_cls_switch = []

    for i in range(n_t-1):
        X_intermediate += [data_out.labels[f'X_{i}']]
        Y_intermediate += [data_out.labels[f'Y_{i}']]
    for i in range(n_t):
        X_cls0 += [data_out.labels[f'X_{i}_0-0']]
        X_cls1 += [data_out.labels[f'X_{i}_1-1']]
        X_cls_switch += [data_out.labels[f'X_{i}_switch']]

    X_in = data_in.labels['X']
    Y_out = data_out.labels['Y']
    Y_sample = data_out.labels['Y_sample']
    X_sample = data_out.labels['X_sample']

    values = {}

    values['X2Y'] = [X_in] + Y_intermediate + [Y_out]
    values['Y2X'] = [Y_sample] + X_intermediate + [X_sample]
    values['Y2X0'] = [Y_sample] + X_cls0
    values['Y2X1'] = [Y_sample] + X_cls1
    values['Y2Xswitch'] = [Y_out] + X_cls_switch

    prng = np.random.RandomState(42)
    choice = prng.choice(len(X_in), size=200, replace=False)

    for k, v in values.items():
        values[k] = [t[choice] for t in v]

    X2Y = values['X2Y'] 
    Y2X = values['Y2X'] 
    label = label[choice]
    label_inv = 1 - label

    color_f = X2Y[0][..., 0] * (X2Y[0][..., 1].max() - X2Y[0][..., 1].min()) + X2Y[0][..., 1]
    color_b = Y2X[0][..., 0] * (Y2X[0][..., 1].max() - Y2X[0][..., 1].min()) + Y2X[0][..., 1]

    ms = ['s', 'o']
    marker_f = [ms[c] for c in label]
    marker_f_switch = [ms[c] for c in label_inv]
    marker_b = 'o'

    w = 10
    f, AX = plt.subplots(len(X2Y), 5,
                         sharex=True, sharey=True,
                         figsize=(w, 0.4*len(X2Y) * w),
                         constrained_layout=True)

    order = ['X2Y', 'Y2Xswitch', 'Y2X', 'Y2X0', 'Y2X1']
    colors = [color_f, color_f, color_b, color_b, color_b]
    markers = [marker_f, marker_f, ['.']*len(X2Y[0]), ['s'] * len(X2Y[0]), ['o']*len(X2Y[0])]

    for i, Ax in enumerate(AX):
        for j, ax in enumerate(Ax):
            if j == 0:
                points = values[order[j]][i]
            else:
                points = values[order[j]][n_t - i]
            color = colors[j]

            marker = markers[j]

            mscatter(points[..., 0], points[..., 1], ax, marker, c=color,
                     ec='k')
            ax.set_aspect(1)
            ax.set_title(f'step {i if j == 0 else n_t - i}')

    f.savefig(os.path.join(root, 'in_n_out_adv.png'), dpi=300)


def plot_callback_cond_simple(root, data_in, data_out, config):
    n_t = retrieve(config, 'model_pars/n_transformers')

    label = data_in.labels['y']

    X_cls0 = data_out.labels[f'X_{n_t - 1}_0-0']
    X_cls1 = data_out.labels[f'X_{n_t - 1}_1-1']
    X_cls_switch = data_out.labels[f'X_{n_t - 1}_switch']

    X_in = data_in.labels['X']
    Y_out = data_out.labels['Y']
    Y_sample = data_out.labels['Y_sample']
    X_sample = data_out.labels['X_sample']

    values = {}

    values['X2Y'] = [X_in] + [Y_out]
    values['Y2X'] = [Y_sample] + [X_sample]
    values['Y2X0'] = [Y_sample] + [X_cls0]
    values['Y2X1'] = [Y_sample] + [X_cls1]
    values['Y2Xswitch'] = [Y_out] + [X_cls_switch]

    prng = np.random.RandomState(42)
    choice = prng.choice(len(X_in), size=200, replace=False)

    for k, v in values.items():
        values[k] = [t[choice] for t in v]

    X2Y = values['X2Y'] 
    Y2X = values['Y2X'] 
    label = label[choice]
    label_inv = 1 - label

    color_f = X2Y[0][..., 0] * (X2Y[0][..., 1].max() - X2Y[0][..., 1].min()) + X2Y[0][..., 1]
    color_b = Y2X[0][..., 0] * (Y2X[0][..., 1].max() - Y2X[0][..., 1].min()) + Y2X[0][..., 1]

    ms = ['s', 'o']
    marker_f = [ms[c] for c in label]
    marker_f_switch = [ms[c] for c in label_inv]
    marker_b = 'o'

    w = 10
    f, AX = plt.subplots(len(X2Y), 5,
                         sharex=True, sharey=True,
                         figsize=(w, 0.4*len(X2Y) * w),
                         constrained_layout=True)

    order = ['X2Y', 'Y2Xswitch', 'Y2X', 'Y2X0', 'Y2X1']
    colors = [color_f, color_f, color_b, color_b, color_b]
    markers = [marker_f, marker_f, marker_f, ['s'] * len(X2Y[0]), ['o']*len(X2Y[0])]

    for i, Ax in enumerate(AX):
        for j, ax in enumerate(Ax):
            if j == 0:
                points = values[order[j]][i]
            else:
                points = values[order[j]][1 - i]
            color = colors[j]

            marker = markers[j]

            mscatter(points[..., 0], points[..., 1], ax, marker, c=color,
                     ec='k')
            ax.set_aspect(1)
            ax.set_title(f'step {0 if j == 0 else n_t}')

    f.savefig(os.path.join(root, 'in_n_out_simple.png'), dpi=300)
