import random
import numpy as np
import torch
from torch import nn
import sys
import os

from model_simp import LeveledRNN
from sort_proc_nn import *
from sort_procedures import ArrayEnv, quick_sort, merge_sort

class Net(nn.Module):
    def __init__(self, dim_in, state_dims, dim_out, array_size, truncate_inputs):
        nn.Module.__init__(self)

        self.emb = InstrEmbedding(dim_in, array_size)
        self.rnn = LeveledRNN(dim_in, dim_out, state_dims, truncate_inputs)
        self.head = InstrHead(dim_out, array_size)

    def get_loss_acc_single(self, seq):
        seq_in = [None]+list(seq)
        seq_target = [x[0] for x in seq]
        seq_emb = torch.stack([
            self.emb(instr, result)
            for instr, result in seq
        ])
        seq_out, active_loss = self.rnn(seq_emb)
        seq_loss, seq_acc = [], []
        for x,target in zip(seq_out, seq_target):
            loss, correct = self.head.get_loss_acc(x, target)
            seq_loss.append(loss)
            seq_acc.append(correct.to(torch.float))
        seq_loss = torch.mean(torch.stack(seq_loss))
        seq_acc = torch.mean(torch.stack(seq_acc))
        return seq_loss, active_loss, seq_acc

    def get_loss_acc_multi(self, seqs): # seq_loss, active_loss, seq_acc
        indicators = zip(*(
            self.get_loss_acc_single(seq)
            for seq in seqs
        ))
        indicators = (
            torch.mean(torch.stack(list(indicator)))
            for indicator in indicators
        )
        return indicators

def generate_seq(array_size, algorithm):
    env = ArrayEnv(np.random.permutation(array_size))
    algorithm(env)
    instructions = env.instructions
    env.reset()
    data = [
        (instr, env.run_instr(instr))
        for instr in instructions
    ]
    return data

def generate_seqs(number, array_size, algorithm):
    return [
        generate_seq(array_size, algorithm)
        for _ in range(number)
    ]

def stats_str(stats):
    metrics = list(map(np.mean, zip(*stats)))
    return "seq_loss {}, active_loss {}, accuracy {}".format(*metrics)

def eval_model(model, eval_seqs, batch_size, epoch):
    stats = []
    model.eval()
    for i in range(0,len(eval_seqs),batch_size): # evaluation
        batch = eval_seqs[i:i+batch_size]
        metrics = model.get_loss_acc_multi(batch)
        stats.append([x.item() for x in metrics])
    print("Evaluation {} : {}".format(epoch, stats_str(stats)))
    sys.stdout.flush()

def train(model, train_seqs, eval_seqs, batch_size, epochs,
          ac_loss_c = 1, save_dir = None, save_each = 10, load_epoch = None):

    optimizer = torch.optim.Adam(model.parameters())
    if load_epoch is not None:
        assert(save_dir is not None)
        print("loading epoch {}".format(load_epoch))
        sys.stdout.flush()
        fname = os.path.join(save_dir, "epoch{}".format(load_epoch))
        model.load_state_dict(torch.load(fname))
        optimizer.load_state_dict(torch.load(fname+"_optimizer"))
        start_epoch = load_epoch
    else: start_epoch = 0

    eval_model(model, eval_seqs, batch_size, start_epoch)

    if save_dir is not None: os.makedirs(save_dir, exist_ok=True)
    train_seqs = list(train_seqs)
    for epoch in range(start_epoch, epochs):
        random.shuffle(train_seqs)
        stats = []
        model.train()
        for i in range(0,len(train_seqs),batch_size): # training
            batch = train_seqs[i:i+batch_size]
            optimizer.zero_grad()
            metrics = tuple(model.get_loss_acc_multi(batch))
            seq_loss, active_loss, seq_acc = metrics
            loss = seq_loss + ac_loss_c*active_loss
            loss.backward()
            optimizer.step()
            stats.append([x.item() for x in metrics])
            print("Training {}, {} : {}".format(
                epoch, min(i+batch_size, len(train_seqs)), stats_str(stats)))
            sys.stdout.flush()
            stats = []
        if stats:
            print("Training {}, {} : {}".format(epoch, len(train_seqs), stats_str(stats)))
            sys.stdout.flush()

        if save_dir is not None and (epoch+1) % save_each == 0:
            fname = os.path.join(save_dir, "epoch{}".format(epoch+1))
            torch.save(model.state_dict(), fname)
            torch.save(optimizer.state_dict(), fname+"_optimizer")

        eval_model(model, eval_seqs, batch_size, epoch+1)

if __name__ == "__main__":
    seed = 42
    array_size = 20
    dim_in = 20
    state_dims = [10]*10
    dim_out = 20
    truncate_inputs = False
    train_num = 200
    eval_num = 20
    batch_size = 10
    epochs = 500
    ac_loss_c = 1
    algorithm = quick_sort

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = Net(dim_in, state_dims, dim_out, array_size, truncate_inputs)
    train_seqs = generate_seqs(train_num, array_size, algorithm)
    eval_seqs = generate_seqs(eval_num, array_size, algorithm)
    print(list(map(len, eval_seqs)))
    exit()

    train(model, train_seqs, eval_seqs, batch_size, epochs, ac_loss_c,
          save_dir = "quicksort_exp1_w")
