import torch


def t2np(tensor):
    return tensor.cpu().detach().numpy()


def np2t(tensor, to=None, cuda=True, to_float=True):
    t = torch.from_numpy(tensor)
    if to is not None:
        t = t.to(to)
    if to_float:
        t = t.float()
    if cuda:
        t = t.cuda()
    return t


class LazyT2OH(torch.nn.Module):
    def __call__(self, some_tensor, nb_digits):
        if not hasattr(self, '_onehot'):
            batch_size = some_tensor.shape[0]
            # One hot encoding buffer that you create out of the loop and just keep reusing
            self._onehot = torch.FloatTensor(batch_size, nb_digits)
            self._nb_digits = nb_digits
        else:
            if self._nb_digits != nb_digits:
                del self._onehot
                self(long_tensor, nb_digits)
        
        self._onehot.zero_()
        if nb_digits > 2:
            self._onehot.scatter_(1, some_tensor, 1)
        else:
            self._onehot[..., 0] = 1. - some_tensor[..., 0]
            self._onehot[..., 1] = some_tensor[..., 0]

        return self._onehot


class Seq2Vec:
    def __init__(self, nt, ne):
        self.nt = nt
        self.ne = ne

    def __call__(self, value, reverse=False):
        if not reverse:
            return self.s2v(value)
        else:
            return self.v2s(value)

    def inverse(self, value):
        return self(value, True)

    def s2v(self, value):
        return value.reshape(-1, self.nt * self.ne)

    def v2s(self, value):
        return value.reshape(-1, self.nt, self.ne)
