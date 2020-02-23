#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

get_ipython().run_line_magic('matplotlib', 'inline')


# # Building Blocks

# In[2]:


class Transformer(nn.Module):
    def __init__(self, conditioner, forward_is_backward=False):
        super().__init__()
        self.C = conditioner
        self.is_forward = not forward_is_backward
        
    def forward(self, value, is_forward=True):
        translation, log_scale = self.C(value)
        
        self.log_scale = log_scale
        self.translation = translation
        self.scale = scale = log_scale.exp()
        
        if is_forward:
            return scale * value + translation
        else:
            return (value - translation) / scale
    
    def inverse(self, value):
        return self(value, is_forward=False)
    
    def inv(self, value):
        return self.inverse(value)


# In[3]:


class FCC(nn.Module):
    def __init__(self, nc, n_hidden=128, n_layers=3):
        super().__init__()
        
        self.nc = nc
        
        layers = []
        layers += [nn.Linear(nc, n_hidden), nn.ReLU()]
        
        for i in range(1, n_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
            
        layers += [nn.Linear(n_hidden, 2*nc)]
        
        self.fn = nn.Sequential(*layers)
        
    def forward(self, value):
        pars = self.fn(value)
        translation, scale = pars[:, :self.nc], pars[:, self.nc:]
        
        return translation, scale


# In[4]:


fcc = FCC(2)
print(fcc)

fcc(torch.from_numpy(np.ones([1, 2])).float())


# In[5]:


T = Transformer(fcc)
print(T)
value = torch.from_numpy(np.ones([1, 2])).float()
print(T(value))
print(T.inv(value))


# # Data

# In[6]:


from sklearn import datasets


# In[7]:


from edflow.data.dataset_mixin import DatasetMixin
from edflow.util import edprint


# In[8]:


class MoonDataset(DatasetMixin):
    def __init__(self, n_samples=1500, noise=0.05):
        self.X, self.y = datasets.make_moons(n_samples=n_samples, noise=noise)
        
        self.append_labels = True
        
    @property
    def labels(self):
        if not hasattr(self, '_labels'):
            self._labels = {}
            self._labels['X'] = self.X
            self._labels['y'] = self.y
        return self._labels
    
    def get_example(self, idx):
        return {}
    
    def __len__(self):
        return len(self.labels['X'])


# In[9]:


class NormalDataset(DatasetMixin):
    def __init__(self, nc, n_samples=1500):
        self.X = np.random.normal(size=[n_samples, nc])
        self.X[:, 0] = self.X[:, 0] / self.X[:, 0].max()
        self.X[:, 1] = self.X[:, 1] / self.X[:, 1].max()
        self.append_labels = True
        
    @property
    def labels(self):
        if not hasattr(self, '_labels'):
            self._labels = {}
            self._labels['X'] = self.X
        return self._labels
    
    def get_example(self, idx):
        return {}
    
    def __len__(self):
        return len(self.labels['X'])


# In[10]:


Moon = MoonDataset()
N = NormalDataset(2)

X = Moon.labels['X']
y = Moon.labels['y']
X_norm = N.labels['X']

f, [ax1, ax2] = plt.subplots(1, 2)

xlim = [-1.3, 2.3]
ylim = [-1.3, 1.3]

ax1.set_aspect(1)
ax2.set_aspect(1)

ax1.scatter(X_norm[:, 0], X_norm[:, 1])
ax2.scatter(X[:, 0], X[:, 1], c=y)

ax1.set_xlim(*xlim)
ax2.set_xlim(*xlim)
ax1.set_ylim(*ylim)
ax2.set_ylim(*ylim)


# In[11]:


edprint(Moon[0])


# # Model(s)

# In[12]:


class TransformerModel(nn.Module):
    def __init__(self, n_transformers=4):
        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(n_transformers):
            C = FCC(2)
            T = Transformer(C)
            self.transformers += [T]

    def forward(self, value, is_forward=True):
        self.conditions = {'translation': [], 'scale': [], 'log_scale': []}
        if is_forward:
            for T in self.transformers:
                value = T(value, is_forward=True)
                self.conditions['translation'] += [T.translation]
                self.conditions['scale'] += [T.scale]
                self.conditions['log_scale'] += [T.log_scale]
        else:
            for T in self.transformers[::-1]:
                value = T(value, is_forward=False)
                self.conditions['translation'] += [T.translation]
                self.conditions['log_scale'] += [T.log_scale]
            
        return value
    
    def inverse(self, value):
        return self(value, is_forward=False)


# In[13]:


TM1 = TransformerModel(1)
print(TM1)


# # Training

# In[14]:


from edflow.iterators.template_iterator import TemplateIterator
from edflow.custom_logging import init_project
from edflow.iterators.batches import make_batches


# In[18]:


class Iterator(TemplateIterator):
    def __init__(self, config, root, model, dataset, **kwargs):
        self.model = model
        
        self.optim = torch.optim.SGD(self.model.parameters(), lr=1, momentum=0.9)
        
        super().__init__(config, root, model, dataset, **kwargs)
        
    def step_op(self, _, labels_, **kwargs):
        X = torch.from_numpy(labels_['X']).float()
        
        Y = self.model(X)
        log_scales = self.model.conditions['log_scale']
        
        log_det_loss = -log_scales[0].sum(-1)
        for log_scale in log_scales[1:]:
            log_det_loss += -log_scale.sum(-1)
        log_det_loss = torch.mean(log_det_loss)
        
        kl_loss = torch.mean(Y ** 2)
        
        loss = kl_loss + log_det_loss
        
        def train_op():
            self.optim.zero_grad()
            
            loss.backward()
            
            self.optim.step()
        
        def eval_op():
            pass
        
        def log_op():
            return {'scalars': {'loss': loss, 'kl': kl_loss, 'det': log_det_loss}}
        
        return {'train_op': train_op, 'eval_op': eval_op, 'log_op': log_op}
    
    def save(self, path):
        pass


# In[19]:


P = init_project('logs', code_root=None, postfix='tflow')

Trainer = Iterator({'test_mode': False}, P.root, TM1, Moon, num_epochs=5)


# In[20]:


Trainer.iterate(make_batches(Moon, batch_size=64, shuffle=True))

del P


# In[ ]:





# In[ ]:




