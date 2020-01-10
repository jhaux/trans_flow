from sklearn import datasets
import numpy as np

from edflow.data.dataset_mixin import DatasetMixin
from edflow.util import edprint, retrieve


class MoonDataset(DatasetMixin):
    def __init__(self, config):
        n_samples = retrieve(config, 'data/nsamples', default=1500)
        noise = retrieve(config, 'data/noise', default=0.05)
        self.X, self.y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
        
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
