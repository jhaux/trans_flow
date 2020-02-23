import torch
import torch.nn as nn
from edflow.util import retrieve
from tflow.util import Seq2Vec


class BehaviorModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enc = BehaviorEncoder(config)
        self.dec = BehaviorDecoder(config)

        self.start_size = retrieve(config, 'model_pars/start_size')

    def forward(self, pose_sequence):
        behavior_encoding = self.enc(pose_sequence)
        start_signal = pose_sequence[:self.start_size]
        generated_sequence = self.dec(
            start_signal,
            behavior_encoding,
            len(pose_sequence)
        )

        return generated_sequence

    
class BehaviorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_layers = retrieve(config, 'model_pars/behavior/num_layers')
        input_size = retrieve(config, 'model_pars/behavior/input_size')
        behavior_size = retrieve(config, 'model_pars/behavior_size')

        self.num_layers = num_layers
        self.behavior_size = behavior_size
        
        self.rnn = nn.GRU(input_size, behavior_size, num_layers=num_layers,
                          bias=True, batch_first=True)

    def forward(self, input_sequence):
        in_shape = input_sequence.shape
        bs, T, K = in_shape
        out, hidden_state = self.rnn(input_sequence)

        behavior_encoding = torch.cat(hidden_state.split(1, dim=0), dim=-1).squeeze(0)

        return behavior_encoding


class BehaviorDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_layers = retrieve(config, 'model_pars/behavior/num_layers')
        output_size = retrieve(config, 'model_pars/behavior/input_size')
        b_const = retrieve(config, 'model_pars/behavior/constant_hidden_state')
        behavior_size = retrieve(config, 'model_pars/behavior_size')

        self.num_layers = num_layers
        self.behavior_size = behavior_size
        self.b_const = b_const
        
        self.rnn = nn.GRU(output_size, behavior_size,
                          num_layers=num_layers, bias=True, batch_first=True)
        self.output_layer = nn.Linear(behavior_size, output_size)

    def forward(self, start_condition, behavior_encoding, output_length):
        '''
        Parameters
        ----------
        start_condition : a number of poses which are given as inputs for the
            first timesteps.
        behavior_encoding : torch.Tensor
            A vector representation of the displayed behavior
        output_length : int
            The number of poses to output
        '''

        output_sequence = []
        hidden = torch.stack(behavior_encoding.split(256, dim=1), dim=0)

        for pose in start_condition.split(1, dim=1):
            result = self.rnn(pose, hidden)
            if self.b_const:
                out, _ = result
            else:
                out, hidden = result

            out = self.output_layer(out)

            output_sequence += [out]

        for t in range(start_condition.shape[1], output_length):
            result = self.rnn(out, hidden)
            if self.b_const:
                out, _ = result
            else:
                out, hidden = result

            out = self.output_layer(out)

            output_sequence += [out]

        output_sequence = torch.cat(output_sequence, dim=1)

        return output_sequence
