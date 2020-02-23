from edflow.iterators.template_iterator import TemplateIterator
from edflow.util import edprint, retrieve
import torch
import os
import numpy as np

import matplotlib.pyplot as plt

from tflow.util import t2np, np2t, LazyT2OH


t2oh = LazyT2OH()


class Iterator(TemplateIterator):
    def __init__(self, config, root, model, datasets, **kwargs):
        self.model = model
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)  #, momentum=0.9)

        self.n_start = retrieve(config, 'model_pars/start_size')
        self.t_offset = retrieve(config, 'model_pars/prediction_offset')
        self.behavior_size = retrieve(config, 'model_pars/behavior_size')

        self.train_stage_1 = retrieve(config, 'training/stage_1')
        self.train_stage_2 = retrieve(config, 'training/stage_2')
        
        super().__init__(config, root, model, datasets, **kwargs)
        
    def step_op(self, _, labels_, **kwargs):
        X = labels_['X']
        X = np2t(X, cuda=False)
        X = self.maybe_cuda(X)

        cls = np2t(labels_['y'][:, 0, None],
                   cuda=False,
                   to_float=False)
        cls = self.maybe_cuda(t2oh(cls, 2))

        bs, T, pose = X.shape

        behavior_encoding = self.model.stage1.enc(X)

        start_condition = X[:, :self.n_start]
        output_length = T - self.t_offset
        X_rec = self.model.stage1.dec(start_condition,
                                      behavior_encoding,
                                      output_length)

        loss = 0
        if self.train_stage_1:
            kl_stage1 = torch.mean(0.5 * torch.sum(behavior_encoding ** 2, dim=-1))

            X_targ = X[:, self.t_offset:]
            rec_loss = torch.mean((X_rec - X_targ) ** 2)

            loss += kl_stage1 + rec_loss
        
        if self.model.stage2.is_cond:
            Y = self.model.stage2(behavior_encoding, cls)
        else:
            Y = self.model.stage2(behavior_encoding)

        if self.train_stage_2:
            log_det_loss = -torch.mean(self.model.stage2.log_det)
            kl_loss = torch.mean(0.5 * torch.sum(Y ** 2, dim=-1))
        
            loss += kl_loss + log_det_loss

        def train_op():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        def eval_op():
            if not self.model.stage2.is_cond:
                Y_inv = self.model.stage2.inv(Y)
            else:
                Y_inv = self.model.stage2.inv(Y, cls)

            X_inv = self.model.stage1.dec(start_condition, Y_inv, output_length)

            ret_d = {'labels': {
                'Y': t2np(Y),
                'Y_inv': t2np(Y_inv),
                'pos_sample': t2np(X_inv)
            }}

            if self.model.stage2.is_cond:
                # Case 3: invert class 
                Y_inv_switch = self.model.stage2.inv(Y, 1 - cls)
                X_inv_switch = self.model.stage1.dec(start_condition, Y_inv_switch, output_length)

                ret_d['labels'][f'Y_inv_switch'] = t2np(X_inv)
                ret_d['labels'][f'X_inv_switch'] = t2np(X_inv)

                ret_d['labels']['cls'] = t2np(cls)
                ret_d['labels']['cls_inv'] = t2np(1 - cls)

            return ret_d
        
        def log_op():
            return {'scalars':
                    {'loss': loss,
                     'stage1/kl': kl_stage1, 'stage1/rec': rec_loss,
                     'stage2/kl': kl_loss, 'stage2/det': log_det_loss}
                    }
        
        return {'train_op': train_op, 'eval_op': eval_op, 'log_op': log_op}
    
    def save(self, path):
        self.model.cpu()
        torch.save(self.model.state_dict(), path)
        if torch.cuda.is_available():
            self.model.cuda()
    
    def restore(self, checkpoint_path):
        self.model.cpu()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        if torch.cuda.is_available():
            self.model.cuda()

    def maybe_cuda(self, t):
        if torch.cuda.is_available():
            t = t.cuda()
        return t


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
    label = data_in.labels['cls']
    label_inv = data_in.labels['cls_inv']

    print(label.mean())
    print(label_inv.mean())

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
    label_inv = label_inv[choice]
    # label_inv = 1 - label

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
    label = data_out.labels['cls'].astype(int).argmax(-1)
    label_inv = data_out.labels['cls_inv'].astype(int).argmax(-1)

    print(np.unique(label))
    print(np.unique(label_inv))
    print(label.mean())
    print(label.shape)
    print(label_inv.mean())

    X_cls_switch = data_out.labels['X_inv_switch']

    X_in = data_in.labels['X']
    Y_out = data_out.labels['Y']

    values = {}

    values['X2Y'] = [X_in] + [Y_out]
    values['Y2X'] = [Y_sample] + [X_sample]
    values['Y2Xswitch'] = [Y_out] + [X_cls_switch]

    prng = np.random.RandomState(42)
    choice = prng.choice(len(X_in), size=200, replace=False)

    for k, v in values.items():
        values[k] = [t[choice] for t in v]

    X2Y = values['X2Y'] 
    Y2X = values['Y2X'] 
    label = label[choice]
    label_inv = label_inv[choice]

    color_f = X2Y[0][..., 0] * (X2Y[0][..., 1].max() - X2Y[0][..., 1].min()) + X2Y[0][..., 1]
    color_b = Y2X[0][..., 0] * (Y2X[0][..., 1].max() - Y2X[0][..., 1].min()) + Y2X[0][..., 1]

    ms = ['s', 'o']
    marker_f = [ms[c] for c in label]
    marker_f_switch = [ms[c] for c in label_inv]
    marker_b = 'o'

    w = 10
    f, AX = plt.subplots(len(X2Y), 3,
                         sharex=True, sharey=True,
                         figsize=(w, 0.4*len(X2Y) * w),
                         constrained_layout=True)

    order = ['X2Y', 'Y2Xswitch', 'Y2X']
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

def mean_cls(root, din, dout, c):
    print(dout.labels['cls'].mean())
    print(dout.labels['cls_inv'].mean())
