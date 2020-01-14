from sklearn import datasets
import numpy as np
import os

from edflow.data.dataset_mixin import DatasetMixin, SubDataset
from edflow.data.believers.sequence import SequenceDataset
from edflow.data.believers.meta import MetaDataset
from edflow.data.believers.meta_view import MetaViewDataset
from edflow.util import edprint, retrieve


class MoonDataset(DatasetMixin):
    def __init__(self, config):
        n_samples = retrieve(config, 'data/nsamples', default=1500)
        noise = retrieve(config, 'data/noise', default=0.05)
        bias = retrieve(config, 'data/bias', default=0)
        self.X, self.y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)

        prng = np.random.RandomState(42)

        if bias > 0:
            choice = self.y == 1
            X_1 = self.X[choice]
            y_1 = self.y[choice]
            
            choice2 = X_1[..., 0] > X_1[..., 0].mean()
            X_1_add = X_1[choice2] 
            X_1_add = X_1_add + prng.normal(0, noise, size=X_1_add.shape)
            y_1_add = y_1[choice2]

            self.X = np.concatenate([self.X, X_1_add])
            self.y = np.concatenate([self.y, y_1_add])
        
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


class BEMoji(MetaDataset):
    def __init__(self, config):
        root = retrieve(config, 'data/root')

        super().__init__(root)

        self.append_labels = True

        self.labels['X'] = self.labels['finl_trajectory']
        self.labels['y'] = self.labels['health_level']

        if not 'seq_id' in self.labels:

            from edflow.data.believers.meta_util import store_label_mmap

            t_s = self.labels['trajectory_seed']
            s_s = self.labels['size']
            h_s = self.labels['health_level']
            i_s = self.labels['identity']

            t = np.unique(t_s)  # 10
            s = np.unique(s_s)  # 4
            h = np.unique(h_s)  # 5
            i = np.unique(i_s)  # 10

            print(t)
            print(s)
            print(h)
            print(i)

            seq_ids = t_s.astype(float)
            print(np.unique(seq_ids).shape)
            seq_ids += s_s.astype(float)
            print(np.unique(seq_ids).shape)
            seq_ids += h_s.astype(float)
            print(np.unique(seq_ids).shape)
            seq_ids += 10**(i_s+3).astype(int)
            print(np.unique(seq_ids).shape)

            assert len(np.unique(seq_ids)) == 10 * 4 * 5 * 10, (len(np.unique(seq_ids)), 10*4*5*10)

            store_label_mmap(seq_ids, os.path.join(root, 'labels'), 'seq_id')

            self.labels['seq_id'] = seq_ids

    def get_example(self, idx):
        return {}


class BEMoji_Seq(DatasetMixin):
    def __init__(self, config):

        B = MetaViewDataset('/home/jhaux/Dr_J/Data/BEmoji/seq_100')
        self.data = B

        del self.labels['image_']

        edprint(self.labels)

        self.labels['X'] = self.labels['finl_trajectory']
        self.labels['y'] = self.labels['health_level']

        self.append_labels = True

    def get_example(self, idx):
        return {}


class BEMoji_Train(DatasetMixin):
    def __init__(self, config):
        B = BEMoji_Seq(config)
        self.data = SubDataset(B, np.arange(int(0.9 * len(B))))


class BEMoji_Valid(DatasetMixin):
    def __init__(self, config):
        B = BEMoji_Seq(config)
        self.data = SubDataset(B, np.arange(int(0.9 * len(B)), int(0.95 * len(B))))


class BEMoji_Test(DatasetMixin):
    def __init__(self, config):
        B = BEMoji_Seq(config)
        self.data = SubDataset(B, np.arange(int(0.95 * len(B)), len(B)))
