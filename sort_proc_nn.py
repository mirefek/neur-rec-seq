import torch
from torch import nn
import torch.nn.functional as F

class InstrEmbedding(nn.Module):
    def __init__(self, output_dim, array_size):
        nn.Module.__init__(self)

        self.array_size = array_size
        self.output_dim = output_dim
        self.scale = max(1,(self.array_size-1))/2

        self.ini_symb = nn.Parameter(torch.Tensor(output_dim))
        self.ini_symb.data.uniform_(-0.5, 0.5)
        self.embs = nn.ModuleList([
            nn.Linear(2, output_dim), # swap
            nn.Linear(2, output_dim), # less_than False
            nn.Linear(2, output_dim), # less_than True
            nn.Linear(1, output_dim), # stack_push
            nn.Linear(1, output_dim), # stack_pop
        ])

    def forward(self, instr, result):
        if instr is None: return self.ini_symb
        t, args = instr
        if t > 1 or (t == 1 and result): t += 1
        x = torch.tensor(args, dtype = torch.float) / self.scale - 1
        x = self.embs[t](x)
        x = F.relu(x)
        return x

class InstrHead(nn.Module):
    
    def __init__(self, input_dim, array_size):
        nn.Module.__init__(self)
        self.array_size = array_size
        self.logit_lin = nn.Linear(input_dim, 4)
        self.index_lin = nn.Linear(input_dim, 2)

    def forward(self, x): # [idim] -> [4], [2]
        return self.logit_lin(x), self.index_lin(x)

    def get_best_ho(self, head_output):
        logits, args_guess = head_output
        t = torch.argmax(logits).item()
        if t in (2,3): args_guess = args_guess[:1]
        args = torch.floor(args_guess+0.5).to(torch.int)
        args = torch.clamp(args, 0, self.array_size-1)
        args = tuple(x.item() for x in args)
        return t,args

    def get_loss_ho(self, head_output, target_instr):
        t, args = target_instr
        args = torch.tensor(args, dtype = torch.float)
        logits, args_guess = head_output
        if t in (2,3): args_guess = args_guess[:1]
        loss = -F.log_softmax(logits, dim = -1)[t] + \
            F.mse_loss(args_guess, args, reduction = 'sum')
        return loss

    def get_best(self, x):
        return self.get_best_ho(self(x))
    def get_loss_acc(self, x, target):
        head_input = self(x)
        correct = torch.tensor(self.get_best_ho(head_input) == target)
        loss = self.get_loss_ho(head_input, target)
        return loss, correct
