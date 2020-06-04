import torch
from general_seq import tm_gen, fw_gen

def pixel(x):
    if isinstance(x, torch.Tensor): x = x.detach().numpy()
    x = x.reshape([-1])
    assert(len(x) > 0)
    x = np.clip(np.floor(x*256), 0, 255)
    if len(x) == 1: return np.tile(x, 3)
    elif len(x) == 2: return np.concatenate([x, [0]])
    elif len(x) == 3: return x
    else: return x[:3]

def make_seq_tm(n):
    return torch.tensor(tm_gen.generate(n))
def make_seq_fw(n):
    return torch.tensor(fw_gen.generate(n))
