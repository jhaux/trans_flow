import torch
import torch.nn as nn

from tflow.models.blocks import Transformer, FCC, FCCC
from edflow.util import retrieve
from tflow.util import t2np, np2t, LazyT2OH, Seq2Vec
t2oh = LazyT2OH()


class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_transformers = retrieve(config, 'model_pars/n_transformers', default=4)
        self.is_cond = retrieve(config, 'model_pars/conditional', default=False)
        ls_ = retrieve(config, 'model_pars/behavior_size')
        num_behave_layers = retrieve(config, 'model_pars/behavior/num_layers')

        self.latent_size = ls = ls_ * num_behave_layers

        self.transformers = nn.ModuleList()
        for i in range(n_transformers):
            C = FCC(ls // 2) if not self.is_cond else FCCC(ls // 2)
            T = Transformer(C, self.is_cond, bool(i % 2))
            self.transformers += [T]

        self.test()

    def forward(self, value, cls=None, is_forward=True):
        self.intermediates = []
        if is_forward:
            self.log_det = 0
            for T in self.transformers:
                value = T(value) if not self.is_cond else T(value, cls)
                self.intermediates += [value]
                self.log_det = self.log_det + T.log_det
        else:
            for T in self.transformers[::-1]:
                value = T.inverse(value) if not self.is_cond else T.inverse(value, cls)
                self.intermediates += [value]
            
        return value
    
    def inverse(self, value, cls=None):
        return self(value, cls, is_forward=False)

    def inv(self, *args, **kwargs):
        return self.inverse(*args, **kwargs)

    def test(self):
        cls = t2oh(torch.ones(10, 1).long(), 2) if self.is_cond else None
        in_val = torch.ones(10, self.latent_size).float()
        in_val.normal_()
        out_val = self(in_val, cls).float()
        rev_val = self.inverse(out_val, cls).float()

        equal = in_val == rev_val
        all_eq = torch.allclose(in_val, rev_val)

        for T in self.transformers:
            T.norm.is_initialized = False

        all_eq = torch.allclose(in_val, rev_val, 1e-5, 1e-6)
        assert all_eq, (in_val, rev_val, in_val - rev_val)

if __name__ == '__main__':
    TM1 = TransformerModel(1)
    print(TM1)
