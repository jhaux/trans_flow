import torch
import torch.nn as nn

from tflow.blocks import Transformer, FCC, FCCC
from edflow.util import retrieve
from tflow.util import t2np, np2t, LazyT2OH
t2oh = LazyT2OH()


class TransformerModel(nn.Module):
    def __init__(self, config):
        n_transformers = retrieve(config, 'model_pars/n_transformers', default=4)
        self.is_cond = retrieve(config, 'model_pars/conditional', default=False)

        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(n_transformers):
            C = FCC(1) if not self.is_cond else FCCC(1)
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
        in_val = torch.ones(10, 2).float()
        out_val = self(in_val, cls).float()
        rev_val = self.inverse(out_val, cls).float()

        equal = in_val == rev_val
        all_eq = torch.allclose(in_val, rev_val)

        assert all_eq, (in_val.data, rev_val.data)

if __name__ == '__main__':
    TM1 = TransformerModel(1)
    print(TM1)