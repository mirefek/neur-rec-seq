import torch
from torch import nn
import torch.nn.functional as F

class HanoiEmbedding(nn.Module):
    def __init__(self, size, use_goal = True,
                 dim = 50, layer_num = 6, conv_r = 2):
        nn.Module.__init__(self)

        self.size = size
        self.dim = dim

        self.use_goal = use_goal
        self.emb_bars = nn.Embedding(3, dim)
        if self.use_goal: self.emb_goal = nn.Embedding(4, dim)

        self.conv_layers = nn.ModuleList(
            nn.Conv1d(dim, dim, 2*conv_r-1, padding = conv_r)
            for _ in range(layer_num)
        )
    def forward(self, states):

        bars = torch.tensor([state['bars'] for state in states])
        emb = self.emb_bars(bars)      # [bs, size, dim]
        if self.use_goal:
            goal = torch.tensor([state['goal'] for state in states])
            goal_1h = F.one_hot(goal[:,0], self.size)*(1+goal[:,1:])
            emb = emb + self.emb_goal(goal_1h)

        x = torch.transpose(emb, 1,2) # [bs, dim, size]
        x = F.relu(x)
        for conv in self.conv_layers:
            x = conv(x)      # [bs, dim, size]
            x = F.relu(x)
        x,_ = torch.max(x, dim = -1) # [bs, dim]
        return x

class BottlesEmbedding(nn.Module):
    def __init__(self, size, use_goal = True,
                 dim = 50, layer_num = 6, conv_r = 2):
        nn.Module.__init__(self)

        self.size = size
        self.dim = dim
        self.use_goal = use_goal

        self.emb_pl = nn.Embedding(2, dim)
        self.emb_bot = nn.Embedding(3, dim)
        self.emb_ba = nn.Embedding(2, dim)
        if self.use_goal: self.emb_goal = nn.Embedding(4, dim)

        self.conv_layers = nn.ModuleList(
            nn.Conv1d(dim, dim, 2*conv_r-1, padding = conv_r)
            for _ in range(layer_num)
        )
    def forward(self, states):

        platter = torch.tensor([state['platter'] for state in states]) # [bs]
        bottles = torch.tensor([state['bottles'] for state in states]) # [bs, size]
        balls = torch.tensor([state['balls'] for state in states])     # [bs, size]
        platter_1h = F.one_hot(platter, self.size) # [bs, size]

        emb = self.emb_pl(platter_1h) # [bs, size, dim]
        emb = emb + self.emb_bot(bottles)
        emb = emb + self.emb_ba(balls)
        if self.use_goal:
            goal = torch.tensor([state['goal'] for state in states])
            goal_1h = F.one_hot(goal[:,0], self.size)*(1+goal[:,1:])
            emb = emb + self.emb_goal(goal_1h)

        x = torch.transpose(emb, 1,2)    # [bs, dim, size]
        x = F.relu(x)
        for conv in self.conv_layers:
            x = conv(x)      # [bs, dim, size]
            x = F.relu(x)
        x,_ = torch.max(x, dim = -1) # [bs, dim]
        return x
