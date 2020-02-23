from tflow.models.behavior import BehaviorModel
from tflow.models.transformer import TransformerModel

import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.stage1 = BehaviorModel(config)
        self.stage2 = TransformerModel(config)
