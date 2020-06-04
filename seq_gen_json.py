import sys
import os
import torch
import torch.optim as optim
from model import SeqGen
from helpers import make_seq_tm, make_seq_fw
from training import train_seq

if __name__ == "__main__":

    import json

    config = json.loads(sys.stdin.read())
    seed = config.pop("seed")
    seq_len = config.pop("seq_len")
    seq_type = config.pop("seq_type")
    truncate_input = config.pop("truncate_input")
    layer_dim = config.pop("layer_dim")
    layer_num = config.pop("layer_num")

    torch.manual_seed(seed)
    if seq_type == 'tm': seq = make_seq_tm(seq_len)
    elif seq_type == 'fw': seq = make_seq_fw(seq_len)
    else: raise Exception("unknown sequence type '{}'".format(seq_type))

    model = SeqGen(2, [layer_dim]*layer_num, truncate_input = truncate_input)

    out = train_seq(model, seq, **config)

    print(json.dumps(out, indent=4))
