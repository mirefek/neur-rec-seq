import os
import sys
import torch

"""
Parameters:
  model:
    SeqGen instance from model.py
  seq:
    one-dimensional tensor of symbols
  steps:
    the number of training steps
  optimizer -- default: {'type' : "Adam"}:
    dict containing a key "type",
    and the other keys corresponding to keyword arguments of the appropriate PyTorch optimizer
  clip_norm:
    either single number for gradient norm clipping,
    or keyword arguments dict for torch.nn.utils.clip_grad_norm_
    no clipping if not set (None)
  verbose, logf -- default: True, sys.stdout
    if verbose is True, continuous loss and mistakes are printed to logf
  save_dir, save_each -- default None, 50:
    if save_dir is set, then the network and optimizer weights
    will be saved to the directory save_dir every "save_each" step
  load:
    if set to a tuple (load_dir, step), it loads the network and optimizer weights
    and continue training from there
"""
def train_seq(model, seq, steps, optimizer = {'type' : "Adam"}, clip_norm = None, verbose = True,
              logf = sys.stdout, save_dir = None, save_each = 50, load = None):
    if save_dir is not None: os.makedirs(save_dir, exist_ok=True)
    model.train()

    parameters = tuple(model.parameters())
    optimizer_args = dict(optimizer)
    optimizer_type = optimizer_args.pop('type')
    optimizer = getattr(torch.optim, optimizer_type)(parameters, **optimizer_args)

    if load is not None:
        load_dir, start_step = load
        fname = os.path.join(load_dir, "step{}".format(start_step))
        model.load_state_dict(torch.load(load_fname))
        optimizer.load_state_dict(torch.load(load_fname+"_optimizer"))
    else: start_step = 0

    out = dict()
    def store_out(label, value, step):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        if step == 0:
            out[label+"_sum"] = value
            out[label+"_min"] = value
        else:
            out[label+"_sum"] += value
            out[label+"_min"] = min(value, out[label+"_min"])
        out[label+"_last"] = value
    def get_out_avg(label):
        out[label+"_avg"] = float(out[label+"_sum"]) / steps
        del out[label+"_sum"]

    for step in range(start_step, start_step + steps):
        optimizer.zero_grad()
        loss, acc = model.get_loss(seq)

        mistakes = len(seq) - acc
        store_out('loss', loss, step)
        store_out('mistakes', mistakes, step)
        if verbose:
            logf.write("{} {} {}\n".format(step, loss.item(), mistakes))
            logf.flush()

        loss.backward()
        if clip_norm is not None:
            if not isinstance(clip_norm, dict): clip_norm = {'max_norm' : clip_norm}
            torch.nn.utils.clip_grad_norm_(parameters, **clip_norm)
        optimizer.step()

        if save_dir is not None and (step+1) % save_each == 0:
            fname = os.path.join(save_dir, "step{}".format(step+1))
            torch.save(model.state_dict(), fname)
            torch.save(optimizer.state_dict(), fname+"_optimizer")

    get_out_avg('loss')
    get_out_avg('mistakes')

    return out
