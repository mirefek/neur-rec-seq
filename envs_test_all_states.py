import numpy as np
from itertools import product
import random
import torch
from torch import nn
import torch.nn.functional as F

from recursive_envs import Hanoi, Bottles
from game_state_embedding import HanoiEmbedding, BottlesEmbedding

def make_hanoi_data(size = 8):
    data = []
    env = Hanoi(size)
    for goal_disc in range(size):
        for bars in product(*[[0,1,2]]*size):
            b = bars[goal_disc]
            for goal_bar in (b+1)%3, (b+2)%3:
                env.bars = list(bars)
                env.goal_disc = goal_disc
                env.goal_bar = goal_bar

                state = env.get_state()
                action = env.expert_action()
                assert(action is not None)
                data.append((state, action))
    return data, env.size, env.action_space.n

def make_bottles_data(size = 7):
    data = []
    env = Bottles(size)
    for platter in range(size):
        bb_options = []
        if platter > 1: bb_options += [[(0,1)]]*(platter-1)
        basic_option = [(0,1), (1,0), (1,1)]
        if platter > 0: bb_options.append(basic_option)
        bb_options.append(basic_option + [(2,0),(2,1)])
        bb_options += [basic_option]*(size-1-platter)
        assert(len(bb_options) == size)
        for bb in product(*bb_options):
            bottles, balls = zip(*bb)
            if platter > 0 and bottles[platter] == 2 and balls[platter-1] == 1:
                continue
            for goal_bottle in range(size):
                p = bottles[goal_bottle]
                for goal_pos in (p+1)%3, (p+2)%3:
                    env.platter = platter
                    env.bottles = np.array(bottles)
                    env.balls = np.array(balls)
                    env.goal_bottle = goal_bottle
                    env.goal_pos = goal_pos

                    state = env.get_state()
                    action = env.expert_action()
                    assert(action is not None)
                    data.append((state, action))
    return data, env.size, env.action_space.n

class Net(nn.Module):
    def __init__(self, size, emb_class, actions, hidden_dim = 100, emb_kwargs = {}):
        nn.Module.__init__(self)

        self.emb = emb_class(size, **emb_kwargs)
        self.hidden_l = nn.Linear(self.emb.dim, hidden_dim)
        self.action_l = nn.Linear(hidden_dim, actions)

    def get_logits(self, states):
        x = self.emb(states)
        x = self.hidden_l(x)
        x = F.relu(x)
        x = self.action_l(x)
        return x

    def get_loss_acc_multi(self, data_pairs):
        states, target_actions = zip(*data_pairs)
        target_actions = torch.tensor(target_actions)
        logits = self.get_logits(states)
        log_probs = F.log_softmax(logits, dim = -1)
        loss = F.nll_loss(log_probs, target_actions)
        guess_actions = torch.argmax(logits, dim = -1)
        accuracy = torch.mean((guess_actions == target_actions).to(torch.float))
        return loss, accuracy

if __name__ == "__main__":
    train_config = {
        'batch_size' : 50,
        'epochs' : 50,
        'train_print_each' : 40,
    }

    from sort_proc_lstm import train

    seed = 42
    train_size = 20000
    test_size = 200

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    

    data, size, actions = make_hanoi_data()

    random.shuffle(data)
    train_data = data[0 : train_size]
    test_data = data[train_size : train_size+test_size]

    model = Net(size, HanoiEmbedding, actions)
    train(model, train_data, test_data, **train_config)
