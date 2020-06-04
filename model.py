import os
import sys
from itertools import islice, chain
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from log_prob import LogVal, LogProb, log_prob_true
from PIL import Image
from collections import deque

from helpers import pixel, make_seq_tm, make_seq_fw

class SeqGenLayer(nn.Module):
    def __init__(self, out_num, input_dim, output_dim):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.s2s = nn.Linear(input_dim, output_dim, bias = False)
        self.s2o = nn.Linear(input_dim, out_num, bias = False)
        self.s2r = nn.Linear(input_dim, 1, bias = False)
        self.o2s = nn.Embedding(out_num, output_dim)
        self.o2o = nn.Embedding(out_num, out_num)
        self.o2r = nn.Embedding(out_num, 1)

    def forward(self, ori_state, ori_out, ori_active, state, out):
        if len(ori_state) > self.input_dim:
            ori_state = ori_state[:self.input_dim]
        state_update = torch.tanh(self.s2s(ori_state) + self.o2s(ori_out))
        out_update = LogVal(F.log_softmax(self.s2o(ori_state) + self.o2o(ori_out), -1))

        next_out = out*ori_active.neg + out_update*ori_active.pos
        next_state = torch.cat([
            state[:self.output_dim]*ori_active.neg_val()
            + state_update*ori_active.pos_val(),
            state[self.output_dim:],
        ])
        replace = LogProb((self.s2r(ori_state) + self.o2r(ori_out)).squeeze(0))
        next_active = replace * ori_active

        return next_state, next_out, next_active

    def parameters_grouped(self):
        r = [self.s2s, self.s2o, self.o2s, self.o2o], [self.s2r, self.o2r]
        r = [list(chain.from_iterable(x.parameters() for x in g)) for g in r]
        return r

class SeqGen(nn.Module):
    def __init__(self, out_num, dims, truncate_input = True):
        nn.Module.__init__(self)
        self.truncate_input = truncate_input
        self.out_num = out_num
        layer_odims = [
            sum(dims[:i])
            for i in range(1, len(dims)+1)
        ]
        # hidden variables
        self.ini_state = nn.ParameterList(
            nn.Parameter(torch.Tensor(dim))
            for dim in dims
        )
        for s in self.ini_state: s.data.uniform_(-0.5, 0.5)
        self.ini_out = nn.Parameter(torch.Tensor(out_num))
        self.ini_out.data.uniform_(0, 1)

        self.layers = nn.ModuleList()
        for odim in layer_odims:
            if truncate_input: idim = odim
            else: idim = layer_odims[-1]
            self.layers.append(SeqGenLayer(out_num, idim, odim))

    def parameters_grouped(self):
        res = []
        for ini,layer in zip(self.ini_state, self.layers):
            a,b = layer.parameters_grouped()
            a.append(ini)
            res += [a, b]
        res[0].append(self.ini_out)
        return res

    def tm_state_dict(self):
        d = {
            "ini_out" : torch.tensor([1, -1]),
        }
        for i,layer in enumerate(self.layers):
            d["ini_state.{}".format(i)] = torch.tensor([-1, -1])

            idim = layer.input_dim
            odim = layer.output_dim
            layer_d = {
                "s2s.weight" : torch.zeros([odim, idim]),
                "o2s.weight" : torch.zeros([self.out_num, odim]),
                "s2o.weight" : torch.zeros([self.out_num, idim]),
                "o2o.weight" : torch.zeros([self.out_num, self.out_num]),
                "s2r.weight" : torch.zeros([1, idim]),
                "o2r.weight" : torch.zeros([self.out_num, 1]),
            }
            layer_d["s2r.weight"][0,odim-1] = 6

            o0 = odim-2
            o1 = o0+1
            ol = [(x, x+1) for x in range(0, o0, 2)]
            sgn = 1-(i%2)*2
            layer_d["o2o.weight"][0] = torch.tensor([-2, 2])*sgn
            layer_d["o2o.weight"][1] = torch.tensor([2, -2])*sgn
            layer_d["o2s.weight"][0] = torch.tensor([2*sgn, -2]*i + [2*sgn, 2])
            layer_d["o2s.weight"][1] = torch.tensor([-2*sgn, -2]*i + [-2*sgn, 2])

            d.update(
                ("layers.{}.{}".format(i, key), value)
                for key, value in layer_d.items()
            )
        return d

    def set_to_tm(self):
        self.load_state_dict(self.tm_state_dict())

    def fw_state_dict(self):
        d = {
            "ini_out" : torch.tensor([1, -1]),
        }

        for i,layer in enumerate(self.layers):
            d["ini_state.{}".format(i)] = torch.tensor([-1, -1])

            idim = layer.input_dim
            odim = layer.output_dim
            layer_d = {
                "s2s.weight" : torch.zeros([odim, idim]),
                "o2s.weight" : torch.zeros([self.out_num, odim]),
                "s2o.weight" : torch.zeros([self.out_num, idim]),
                "o2o.weight" : torch.zeros([self.out_num, self.out_num]),
                "s2r.weight" : torch.zeros([1, idim]),
                "o2r.weight" : torch.zeros([self.out_num, 1]),
            }
            layer_d["s2r.weight"][0,odim-1] = 6

            if i == 0:
                out = torch.tensor([-2, 2])
                state = torch.tensor([2, 2])
            else:
                out = torch.tensor([2, -2])
                state = torch.tensor([-2, -2] * (i-1) + [-2, 2, 2, 2])
            layer_d["o2o.weight"][0] = out
            layer_d["o2o.weight"][1] = out
            layer_d["o2s.weight"][0] = state
            layer_d["o2s.weight"][1] = state

            d.update(
                ("layers.{}.{}".format(i, key), value)
                for key, value in layer_d.items()
            )
        return d

    def set_to_fw(self):
        self.load_state_dict(self.fw_state_dict())

    def generate(self, real_seq = None, get_process_img = False):
        if real_seq is not None: real_seq = iter(real_seq)
        state = torch.cat(tuple(self.ini_state))
        out = LogVal(self.ini_out)
        while True:
            exact_out = torch.argmax(out.lval)
            if not get_process_img: yield out.lval, exact_out
            if real_seq is not None: exact_out = next(real_seq)

            out = LogVal(torch.zeros(self.out_num))
            ori_state = state
            active = log_prob_true()
            actives = []
            for layer in self.layers:
                state, out, active = layer(ori_state, exact_out, active, state, out)
                actives.append(active)

            if get_process_img:
                img_col1 = [exact_out]
                img_col2 = [out.val()[1]]
                for i, active in enumerate(actives):
                    img_col1.append(ori_state[i*2:(i+1)*2])
                    img_col2.append(state[i*2:(i+1)*2])
                    active = active.pos_val()
                    img_col1.append(active)
                    img_col2.append(active)
                img_col1 = map(pixel, img_col1)
                img_col2 = map(pixel, img_col2)
                yield img_col1
                yield img_col2

    def generate_n(self, n, *args, **kwargs):
        return tuple(map(list, zip(*islice(self.generate(*args, **kwargs), n))))

    def get_img_process(self, n, real_seq = None, fname = None):
        img = list(self.generate_n(2*n, real_seq = real_seq, get_process_img = True))
        img.reverse()
        img = np.array(img, dtype = np.uint8)
        img = Image.fromarray(img, mode = 'RGB')

        if fname is not None: img.save(fname)
        return img

    def get_loss(self, seq):
        n = len(seq)
        logits, out = map(torch.stack, self.generate_n(n, real_seq = seq))
        correct = torch.sum(out == seq).item()
        loss = F.cross_entropy(logits, seq)
        return loss, correct

def check_explicit(n = 400, layers = 15):
    for truncate_input in (True, False):
        gen_tm = SeqGen(2, [2]*layers, truncate_input = truncate_input)
        gen_tm.set_to_tm()
        seq_tm = torch.tensor(gen_tm.generate_n(n)[1])
        print((seq_tm == make_seq_tm(n)).all())

        gen_fw = SeqGen(2, [2]*layers, truncate_input = truncate_input)
        gen_fw.set_to_fw()
        seq_fw = torch.tensor(gen_fw.generate_n(n)[1])
        print((seq_fw == make_seq_fw(n)).all())

def save_imgs(model, seq, load_dir, save_dir, steps_it, n = 400, layers = 15):
    seq_gen = SeqGen(2, [2]*layers)
    for step in steps_it:
        load_fname = os.path.join(load_dir, "step{}".format(step))
        save_fname = os.path.join(save_dir, "step{:04}".format(step))
        print(load_dir, i)
        seq_gen.load_state_dict(torch.load(load_fname))
        seq_gen.get_img_process(n, real_seq = seq, fname = save_fname)

if __name__ == "__main__":
    torch.manual_seed(42)

    check_explicit()
