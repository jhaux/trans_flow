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
    def __call__(self, long_tensor, nb_digits):
        if not hasattr(self, '_onehot'):
            batch_size = long_tensor.shape[0]
            # One hot encoding buffer that you create out of the loop and just keep reusing
            self._onehot = torch.FloatTensor(batch_size, nb_digits)
            self._nb_digits = nb_digits
        else:
            if self._nb_digits != nb_digits:
                del self._onehot
                self(long_tensor, nb_digits)
        
        self._onehot.zero_()
        self._onehot.scatter_(1, long_tensor, 1)

        return self._onehot
