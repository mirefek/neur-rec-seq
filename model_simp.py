import numpy as np
import torch
from torch import nn
from log_prob import LogProb, log_prob_true

class Level(nn.Module):
    def __init__(self, input_dim, output_dim):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.s2s = nn.Linear(input_dim, output_dim)
        self.s2r = nn.Linear(input_dim, 1)

    def forward(self, in_data, state, active): # [bs?, idim], [bs?, odim], [bs?]
        if in_data.shape[-1] > self.input_dim:
            in_data = ori_state[..., :self.input_dim] # [bs?, idim]
        state_update = torch.tanh(self.s2s(in_data)) # [bs?, odim]

        state_dim = state.shape[-1]
        assert state_dim >= self.output_dim
        if state_dim == self.output_dim: # next_state: [bs?, sdim]
            next_state = state*active.neg_val() + state_update*active.pos_val()
        else:
            next_state = torch.cat([
                state[..., :self.output_dim]*active.neg_val()
                + state_update*active.pos_val(),
                state[..., self.output_dim:],
            ], dim = -1)
        next_active = active * LogProb(self.s2r(in_data)) # LogProb(shape=[bs?])

        return next_state, next_active # [bs?, odim], LogProb(shape=[bs?])

class LeveledNet(nn.Module):
    def __init__(self, input_dims, output_dims, truncate_input = True):
        nn.Module.__init__(self)
        self.truncate_input = truncate_input
        self.depth = len(input_dims)
        assert len(output_dims) == self.depth

        self.input_dim = sum(input_dims)
        self.output_dim = sum(output_dims)

        if truncate_input:
            level_idims = np.cumsum(input_dims)
        else: level_idims = self.depth * [self.input_dim]
        level_odims = np.cumsum(output_dims)

        self.levels = nn.ModuleList(
            Level(idim, odim)
            for idim, odim in zip(level_idims, level_odims)
        )

    def forward(self, in_data, x, active = None): # [bs?, idim], [bs?, odim], LogProb(shape=[bs?])
        if active is None: active = log_prob_true()
        for level in self.levels:
            x, active = level(in_data, x, active)

        return x, active # [bs?, odim], LogProb(shape=[bs?])

class LeveledRNN(nn.Module):

    def __init__(self, input_dim, output_dim, dims, truncate_input):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = sum(dims)
        self.ini_state = nn.Parameter(torch.Tensor(self.state_dim))
        self.ini_state.data.uniform_(-0.5, 0.5)
        self.lnet = LeveledNet([input_dim]+dims, [output_dim]+dims, truncate_input)

    def rnn_step(self, in_data, state): # [bs?, idim], [bs?, sdim]
        assert in_data.shape[-1] == self.input_dim
        assert state.shape[-1] == self.state_dim
        lev_in_data = torch.cat([in_data, state], dim = -1) # [bs?, idim + sdim]
        lev_x = torch.cat([
            torch.zeros(in_data.shape[:-1]+(self.output_dim,)),
            state,
        ], dim = -1) # [bs?, odim + sdim]
        lev_out_data, active = self.lnet(lev_in_data, lev_x) # [bs?, odim + sdim], LogProb(shape=[bs?])
        out_data = lev_out_data[..., :self.output_dim] # [bs?, odim]
        state = lev_out_data[..., self.output_dim:] # [bs?, sdim]

        return out_data, state, active # [bs?, odim], [bs?, sdim], LogProb(shape=[bs?])

    def forward(self, seq): # [seq+1, idim]
        state = self.ini_state

        active_l = []
        out_data_l = []
        for in_data in seq:
            out_data, state, active = self.rnn_step(in_data, state)
            active_l.append(active)
            out_data_l.append(out_data)

        active_loss = -torch.mean(torch.stack(
            [active.pos.lval for active in active_l[:-1]]
            + [active_l[-1].neg.lval]
        ))
        out_data = torch.stack(out_data_l[:-1]) # [seq, odim]

        return out_data, active_loss # [seq, odim], []
