#!/usr/bin/python3

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class ArraySorting(gym.Env):
    """
    Description:
        Environment with a hidden state.
        The objective is to sort an array and to be sure that the array is indeed sorted.
        The agent only sees the last performed action, possibly with its boolean outcome.
    Observation: MultiDiscrete([4, size, size]),
        The first value encodes the last observation type,
        the other two are indices i,j of the array:
        Type   Meaning
          0  : initializaction of the array a[i] ... a[j]
          1  : swap of a[i], a[j]
          2  : a[i] < a[j]
          3  : a[j] > a[j]
    Actions:
        Type: MultiDiscrete(2, size, size)
        The first value encodes the action type second two indices i,j of the array:
        Type   Meaning
          0  : swap of a[i], a[j]
          1  : compare a[i], a[j]
    Reward and termination:
        Once the array is sorted and all pairs of consecutive elements were compared,
        the environment terminates with reward 1, otherwise, the reward is zero.
    Starting State:
        A random segment of the array is initialized to a random permutation.
        the actions are restricted to that segment.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }
    def __init__(self, max_size = 50):
        self.max_size = max_size
        self.action_space = spaces.MultiDiscrete([2, max_size, max_size])
        self.observation_space = spaces.MultiDiscrete([4, max_size, max_size])
        self.seed()
        self.viewer = None
        self._expert_data = None
        self.expert_proposal = None
        self.last_action = None
        self.a = None
        self.checked = None
        self.size = None
        self.min_index = None
        self.max_index = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, goal_min = 0, goal_limit = None):
        self.a = np.zeros(self.max_size, dtype = int)
        self.size = self.np_random.randint(2, self.max_size+1)
        self.min_index = self.np_random.randint(0, self.max_size - self.size + 1)
        self.max_index = self.min_index + self.size-1
        self.a[self.min_index : self.max_index+1] = self.np_random.permutation(self.size)
        self.checked = np.zeros(self.size-1)

        self.last_action = 0, self.min_index, self.max_index
        self.expert_reset()

        return self.get_state()

    def step(self, action):
        action = np.array(action)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        assert self.indices_valid(action[1:]), err_msg

        expert_aligned = (action == self.expert_proposal).all()
        t, i, j = action
        if t == 0:
            self.a[i], self.a[j] = self.a[j], self.a[i]
            self.last_action = 1, i, j
            if not expert_aligned: self.expert_reset()
        else:
            ai, aj = (self.a[x] for x in (i,j))
            if ai < aj: res = 2
            else: res = 3
            self.last_action = res, i, j
            if abs(ai - aj) == 1: self.checked[min(ai,aj)] = True

        if expert_aligned: self._expert_next()

        done = self.goal_accomplished()
        reward = float(done)
        return self.get_state(), reward, done, {}

    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def goal_accomplished(self):
        return self.checked.all() and\
            (np.diff(self.a[self.min_index : self.max_index+1]) == 1).all()
    def get_state(self):
        return np.array(self.last_action)
    def indices_valid(self, indices):
        return ((self.min_index <= indices) & (indices <= self.max_index)).all()

    def action_to_str(self, action):
        ai, aj = (self.a[x] for x in action[1:])
        i, j = (x - self.min_index for x in action[1:])
        args = i,ai,j,aj
        if action[0] == 0: return "swap a[{}] (= {}) <-> a[{}] (= {})".format(*args)
        else: return "compare a[{}] (= {}) with a[{}] (= {})".format(*args)

    def expert_action(self):
        if self.expert_proposal[0] < 0: return None
        return np.array(self.expert_proposal)
    def expert_reset(self):
        self._expert_data = [self.min_index,self.max_index], [(self.min_index, self.max_index)]
        self.expert_proposal = 1, self.min_index, self.min_index+1
    def _expert_next(self):
        t = self.last_action[0]
        cur_bounds, stack = self._expert_data
        i,j = cur_bounds
        next_segment = False
        if t == 1: # swap i j
            if i < j: self.expert_proposal = 1,i,i+1
            else: next_segment = True
        elif t == 2: # a[i] < a[j]
            if i+1 == j: next_segment = True
            else:
                self.expert_proposal = 0,i+1,j
                cur_bounds[1] -= 1
        elif t == 3: # a[i] > a[j]
            self.expert_proposal = 0,i,i+1
            cur_bounds[0] += 1

        if next_segment:
            i0, i1 = stack.pop()
            #print(np.array([i0,i,i1])-self.min_index)
            if i1 - i > 1: stack.append((i+1, i1))
            if i - i0 > 1: stack.append((i0, i-1))
            if not stack: self.expert_proposal = -1,-1,-1
            else:
                cur_bounds[:] = list(stack[-1])
                self.expert_proposal = 1, cur_bounds[0], cur_bounds[0]+1

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        h_offset = 50
        v_offset = 35
        rel_bar_dist = 0.3
        rel_conn_height = 0.3
        rel_min_height = 3
        observation_height = 10
        observation_dist = 10

        unchecked_color = 0,0,0
        checked_color = 0.5,0.5,0
        checked_placed_color = 0,1,0
        unactive_bar_color = 0.4,0.4,0.4
        active_bar_colors = [
            (1,0,0), (0,0,1),
        ]
        observation_colors = [
            (0.7, 0.7, 0.7), # init
            (0.8, 0.8, 0),   # swap
        ] + list(reversed(active_bar_colors))

        def rect_coor(l, r, t, b):
            return [(l, b), (l, t), (r, t), (r, b)]

        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self._view_size = 0
            self._observation_view = rendering.FilledPolygon(rect_coor(0,1,0,1))

        x_scale = (screen_width-2*h_offset) / (self.size - rel_bar_dist)
        x_shift = h_offset + x_scale * (1-rel_bar_dist)/2
        y_scale = (screen_height-2*v_offset) / (self.size + rel_min_height - 1)
        y_shift = v_offset

        if self._view_size != self.size: # update number of bars
            del self.viewer.geoms[:]

            self._bars_view = []
            self._bars_trans = []
            basic_trans = rendering.Transform(
                translation = (x_shift, y_shift),
                scale = (x_scale, y_scale),
            )
            self._check_view = []
            self._bars_trans = []
            self._bars_view = []
            for i in range(self.size):
                x = (1-rel_bar_dist)/2
                bar = rendering.FilledPolygon(rect_coor(-x,x,0,i+rel_min_height))
                self._bars_view.append(bar)
                objs = [bar]
                if i > 0:
                    conn1 = rendering.FilledPolygon(rect_coor(-0.5,0,0,rel_conn_height))
                    objs.append(conn1)
                    self._check_view.append((conn2, conn1))
                if i < self.size-1:
                    conn2 = rendering.FilledPolygon(rect_coor(0,0.5,0,rel_conn_height))
                    objs.append(conn2)

                trans = rendering.Transform()
                self._bars_trans.append(trans)
                for obj in objs:
                    self.viewer.add_geom(obj)
                    obj.add_attr(trans)
                    obj.add_attr(basic_trans)

            self.viewer.add_geom(self._observation_view)

        # update bars
        for i,b in enumerate(self.a[self.min_index : self.max_index+1]):
            bar = self._bars_view[b]
            bar.set_color(*unactive_bar_color)
            trans = self._bars_trans[b]
            trans.set_translation(i, 0)
        # update active bars
        for i,color in zip(self.last_action[1:], active_bar_colors):
            bar = self._bars_view[self.a[i]]
            bar.set_color(*color)

        # update checked connections
        b_to_i = np.zeros(self.size, dtype = int)
        b_to_i[self.a[self.min_index : self.max_index+1]] = np.arange(self.size)
        for b,(checked, objs) in enumerate(zip(self.checked, self._check_view)):
            if not checked: color = unchecked_color
            elif b_to_i[b] + 1 != b_to_i[b+1]: color = checked_color
            else: color = checked_placed_color
            for obj in objs: obj.set_color(*color)

        # update observed segment
        lr = np.array(self.last_action[1:], dtype = float) - self.min_index
        lr += np.array([-1,1]) * ((1-rel_bar_dist)/2)
        l,r = x_shift + lr*x_scale
        u = v_offset-observation_dist
        d = u - observation_height
        color = observation_colors[self.last_action[0]]
        self._observation_view.v = rect_coor(l,r,u,d)
        self._observation_view.set_color(*color)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


if __name__ == "__main__":

    from interactive import run_interactive
    run_interactive(ArraySorting(), {})
