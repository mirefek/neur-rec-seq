import sys
import os
import torch
import torch.optim as optim
from model import SeqGen
from helpers import make_seq_tm, make_seq_fw
from training import train_seq

if __name__ == "__main__":
    import argparse

    cmd_parser = argparse.ArgumentParser(prog='seq_gen_cmd',
                                         description='Training of self repetitive sequences',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cmd_parser.add_argument("--layer_dim", default=2, type=int, help="Dimension of a layer")
    cmd_parser.add_argument("--layer_num", default=15, type=int, help="Number of layers")
    cmd_parser.add_argument("--seq_len", default=400, type=int, help="Length of the trained sequence")
    cmd_parser.add_argument("--seed", default=42, type=int, help="Random seed")
    cmd_parser.add_argument("--seq_type", default='tm', type=str, help="'tm' (Thue-Morse) or 'fw' (Fibonacci Word)")
    cmd_parser.add_argument('--no_truncate', dest='truncate_input', action='store_false',
                            help = "uses all layer states for getting the output of a given layer")
    cmd_parser.set_defaults(truncate_input=False)
    cmd_parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    cmd_parser.add_argument("--steps", default=5000, type=int, help="number of training steps")
    cmd_parser.add_argument("--max_norm", default=0.2, type=float, help="maximal norm for gradient clipping")
    cmd_parser.add_argument("--save_each", default=50, type=int, help="every 'save_each' step, the network will be saved")

    config = cmd_parser.parse_args()

    torch.manual_seed(config.seed)
    if config.seq_type == 'tm': seq = make_seq_tm(config.seq_len)
    elif config.seq_type == 'fw': seq = make_seq_fw(config.seq_len)
    else: raise Exception("unknown sequence type '{}'".format(config.seq_type))

    model = SeqGen(2, [config.layer_dim]*config.layer_num, truncate_input = config.truncate_input)

    setup_code = "{}_ti{}_lr{}_seed{}_clip{}".format(config.seq_type, int(config.truncate_input), config.lr, config.seed, config.max_norm)
    weights_dir = setup_code+"_w"
    if config.save_each == 0: weights_dir = None
    log_fname = setup_code+".log"
    with open(log_fname, 'a') as logf:
        train_seq(model, seq, config.steps, optimizer = {'type' : "Adam", 'lr' : config.lr},
                  logf = logf, clip_norm = config.max_norm, save_dir = weights_dir, save_each = config.save_each)
