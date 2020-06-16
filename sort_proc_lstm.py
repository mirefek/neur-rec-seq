import random
import numpy as np
import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence
import sort_procedures
from sort_proc_nn import InstrEmbedding, InstrHead

class Net(nn.Module):
    def __init__(self, dim_in, dim_out, array_size, rnn):
        nn.Module.__init__(self)

        rnn_args = dict(rnn)
        rnn_type = rnn_args.pop('type')
        self.emb = InstrEmbedding(dim_in, array_size)
        self.rnn = getattr(nn, rnn_type)(dim_in, dim_out, **rnn_args)
        self.head = InstrHead(dim_out, array_size)

    def get_loss_acc_single(self, seq):
        seq_in = [None]+list(seq[:-1])
        seq_target = [x[0] for x in seq]
        seq_emb = torch.stack([
            self.emb(instr, result)
            for instr, result in seq
        ])
        rnn_in = torch.unsqueeze(seq_emb, 1)
        rnn_out, _ = self.rnn(rnn_in)
        seq_out = torch.squeeze(rnn_out, 1)
        seq_loss, seq_acc = [], []
        for x,target in zip(seq_out, seq_target):
            loss, correct = self.head.get_loss_acc(x, target)
            seq_loss.append(loss)
            seq_acc.append(correct.to(torch.float))
        seq_loss = torch.mean(torch.stack(seq_loss))
        seq_acc = torch.mean(torch.stack(seq_acc))
        return seq_loss, seq_acc

    def get_loss_acc_multi(self, seqs): # seq_loss, active_loss, seq_acc
        metrics = zip(*(
            self.get_loss_acc_single(seq)
            for seq in seqs
        ))
        metrics = (
            torch.mean(torch.stack(list(metric)))
            for metric in metrics
        )
        return metrics

def generate_seq(array_size, algorithm):
    env = sort_procedures.ArrayEnv(np.random.permutation(array_size))
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
    return "seq_loss {}, accuracy {}".format(*metrics)

def eval_model(model, eval_seqs, batch_size, epoch):
    batch_size = np.gcd(len(eval_seqs), batch_size)
    stats = []
    model.eval()
    for i in range(0,len(eval_seqs),batch_size): # evaluation
        batch = eval_seqs[i:i+batch_size]
        metrics = model.get_loss_acc_multi(batch)
        stats.append([x.item() for x in metrics])
    print("Evaluation {} : {}".format(epoch, stats_str(stats)))
    sys.stdout.flush()

def train(model, train_seqs, eval_seqs, batch_size, epochs,
          optimizer = {'type' : 'Adam'}, train_print_each = 1,
          save_dir = None, save_each = 100, load_epoch = None):

    assert(len(train_seqs) % batch_size == 0)
    parameters = tuple(model.parameters())
    optimizer_args = dict(optimizer)
    optimizer_type = optimizer_args.pop('type')
    optimizer = getattr(torch.optim, optimizer_type)(parameters, **optimizer_args)

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
            loss, seq_acc = metrics
            loss.backward()
            optimizer.step()
            stats.append([x.item() for x in metrics])
            if (i//batch_size + 1) % train_print_each == 0:
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

def generate_data(seed, array_size, train_num, eval_num, algorithm):
    np.random.seed(seed)
    algorithm = getattr(sort_procedures, algorithm)
    train_seqs = generate_seqs(train_num, array_size, algorithm)
    eval_seqs = generate_seqs(eval_num, array_size, algorithm)
    return train_seqs, eval_seqs

if __name__ == "__main__":
    config = {
        'main_seed' : 42,
        'array_size' : 20,
        'data' : {
            'seed' : 42,
            'train_num' : 200,
            'eval_num' : 20,
            'algorithm' : "quick_sort",
        },
        'model' : {
            'dim_in' : 20,
            'dim_out' : 100,
            'rnn' : {
                'type' : 'LSTM',
                'num_layers' : 1,
            },
        },
        'train' : {
            'batch_size' : 10,
            'epochs' : 500,
            'optimizer' : {
                'type' : 'Adam'
            },
        },
    }

    array_size = config['array_size']
    train_seqs, eval_seqs = generate_data(array_size = array_size, **config['data'])

    seed = config['main_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    model = Net(array_size = array_size, **config['model'])

    train(model, train_seqs, eval_seqs, **config['train'])
