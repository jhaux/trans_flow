import numpy as np
import torch
import torch.nn as nn

from edflow.util import retrieve
from tflow.util import t2np, np2t, LazyT2OH
t2oh = LazyT2OH()


class Norm(nn.Module):
    def __init__(self, num_elements):
        super().__init__()
        self.nm = num_elements

        self.trans = nn.Parameter(torch.zeros(1, self.nm), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1, self.nm), requires_grad=True)

        self.is_initialized = False

        self.test()

    def forward(self, value, is_forward=True):
        if not self.is_initialized:
            mu = torch.mean(value, 0)
            sig = torch.std(value, 0)

            self.trans.data.copy_(-mu)
            self.scale.data.copy_(1. / (sig + 1e-6))

            self.is_initialized = True

        if is_forward:
            value = self.scale * (value + self.trans)
            self.log_det = torch.log(torch.abs(self.scale)).sum()
            self.log_det = torch.ones(value.shape[0]).to(self.log_det) * self.log_det

        else:
            value = value / self.scale - self.trans

        return value

    def inverse(self, value):
        return self(value, is_forward=False)

    def test(self):
        in_val = torch.ones(10, self.nm).float()
        in_val.normal_()
        out_val = self(in_val)
        rev_val = self.inverse(out_val).float()

        self.is_initialized = False

        all_eq = torch.allclose(in_val, rev_val, 1e-5, 1e-7)
        assert all_eq, (in_val, rev_val, in_val - rev_val)


class Transformer(nn.Module):
    '''Implements a coupling block'''
    def __init__(self, conditioner, is_cond, inverse_mask=False):
        super().__init__()
        self.C = conditioner
        self.inverse_mask = inverse_mask

        self.is_cond = is_cond

        self.norm = Norm(self.C.nc * 2)

        self.test()

    def forward(self, value, cls=None, is_forward=True):

        if is_forward:
            value = self.norm(value)

        if self.inverse_mask:
            val_id, val_to_change = value.chunk(2, dim=1)
        else:
            val_to_change, val_id = value.chunk(2, dim=1)

        if cls is None:
            translation, log_scale = self.C(val_id)
        else:
            translation, log_scale = self.C(val_id, cls)
        scale = log_scale.exp()
        
        if is_forward:
            val_to_change = scale * val_to_change + translation
            self.log_det = log_scale.sum(-1) + self.norm.log_det
        else:
            val_to_change = (val_to_change - translation) / scale

        if self.inverse_mask:
            value = torch.cat([val_id, val_to_change], dim=1)
        else:
            value = torch.cat([val_to_change, val_id], dim=1)

        if not is_forward:
            value = self.norm.inverse(value)

        return value
    
    def inverse(self, value, cls=None):
        return self(value, cls, is_forward=False)
    
    def inv(self, value, cls=None):
        return self.inverse(value, cls)

    def test(self):
        cls = t2oh(torch.ones(10, 1).long(), 2) if self.is_cond else None
        in_val = torch.ones(10, 2 * self.C.nc).float()
        in_val.normal_()
        out_val = self(in_val, cls).float()
        rev_val = self.inverse(out_val, cls).float()

        all_eq = torch.allclose(in_val, rev_val, 1e-5, 1e-7)

        self.norm.is_initialized = False

        assert all_eq, (in_val.data, rev_val.data, torch.all(in_val == rev_val))


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

        scale = nn.functional.tanh(scale)
        
        return translation, scale


class FCCC(nn.Module):
    def __init__(self, nc, n_hidden=128, n_layers=3):
        super().__init__()
        
        self.nc = nc

        self.cond = nn.Sequential(
            nn.Linear(2, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU()
        )
        
        layers = []
        layers += [nn.Linear(nc, n_hidden), nn.ReLU()]
        
        for i in range(1, n_layers-1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
            
        layers += [nn.Linear(n_hidden, 2*nc)]
        
        self.fn = nn.ModuleList(layers)
        
    def forward(self, value, cls):
        cond = self.cond(cls)
        pars = value
        for i, l in enumerate(self.fn):
            if i > 0:
                pars = l(pars + cond)
            else:
                pars = l(pars)
        translation, scale = pars[:, :self.nc], pars[:, self.nc:]

        scale = nn.functional.tanh(scale)
        
        return translation, scale


if __name__ == '__main__':
    fcc = FCC(2)
    print(fcc)
    
    value = fcc(torch.from_numpy(np.ones([1, 2])).float())
    print(value)
    
    T = Transformer(fcc)
    print(T)
    value = torch.from_numpy(np.ones([1, 2])).float()
    print(T(value))
    print(T.inv(value))
