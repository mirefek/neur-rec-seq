#!/usr/bin/python3

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class Hanoi(gym.Env):
    """
    Description:
        There are discs 0, ..., n-1 on three bars 0,1,2 in increasing order from top to bottom.
        In one step, it is allowed to put one disc from bar to another whole preserving
        the increasing order on bars.
        The agent is supposed to accomplish an goal of the form:
        "Get the disc number d to the bar number b."
        By default, n = 8 (the size keyword argument)
    Observation:
        Type: {
           'bars' : MultiDiscrete(size * [3]),
           'goal' : MultiDiscrete([size, 3]),
        }
        bars: for every disc, the bar on which it is located
        goal: (dics, bar)
    Actions:
        Type: Discrete(6)
        Every action corresponds to putting the top disc from a bar i to a bar j.
        Num   Bars
        0     0 -> 1
        1     0 -> 2
        2     1 -> 2
        3     1 -> 0
        4     2 -> 0
        5     2 -> 1
        If the move is invalid, no action is made.
    Reward and termination:
        Once the goal is accomplished, the game terminates with reward 1,
        otherwise, the reward is zero.
    Starting State:
        Every disc is on a random bar, the goal is assigned randomly
        such that it is not accomplished in the start.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 3
    }

    def __init__(self, size = 8):
        self.size = size
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            'bars' : spaces.MultiDiscrete(size * [3]),
            'goal' : spaces.MultiDiscrete([size, 3]),
        })
        self.seed()
        self.viewer = None
        self.bars = None
        self.goal_disc = None
        self.goal_bar = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, goal_min = 0, goal_limit = None):
        if goal_limit is None: goal_limit = self.size
        assert(goal_min >= 0 and goal_limit <= self.size and goal_min < goal_limit)
        self.bars = list(self.np_random.randint(3, size=self.size))
        self.goal_disc = self.np_random.randint(goal_min, goal_limit)
        self.goal_bar = (self.bars[self.goal_disc] + 1 + self.np_random.randint(2)) % 3

        return self.get_state()

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        b1 = action // 2
        b2 = (b1 + 1 + action % 2) % 3
        if (b1, b2) in self.available_actions_b():
            i = self.bars.index(b1)
            self.bars[i] = b2

        done = self.goal_accomplished()
        reward = float(done)
        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        
        screen_width = 700
        screen_height = 300
        bar_width = 20
        bar_height = screen_height - 50
        disc_offset = 10
        disc_width_max = screen_width * 0.31
        disc_width_min = 0.3 * disc_width_max
        disc_height = bar_height / self.size - disc_offset

        def rect_coor(w,h):
            l, r, t, b = -w/2, w/2, h/2, -h/2
            return [(l, b), (l, t), (r, t), (r, b)]
        def bar_x(b):
            return np.interp(b, [-0.5, 2.5], [0, screen_width])

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # add bars
            self._bars_view = []
            for i in range(3):
                bar = rendering.FilledPolygon(rect_coor(bar_width, bar_height))
                bar.add_attr(rendering.Transform(translation=(bar_x(i), screen_height/2)))
                self.viewer.add_geom(bar)
                self._bars_view.append(bar)

            # add discs, indexed from bottom to top
            self._discs_view_trans = []
            self._discs_view = []
            for i in range(self.size):
                disc_width = np.interp(i, [0,self.size-1], [disc_width_max, disc_width_min])
                disc = rendering.FilledPolygon(rect_coor(disc_width, disc_height))
                disc_view_trans = rendering.Transform()
                self._discs_view_trans.append(disc_view_trans)
                disc.add_attr(disc_view_trans)
                self.viewer.add_geom(disc)
                self._discs_view.append(disc)

        if self.bars is None: return None

        # colorize goal_bar
        for i, bar in enumerate(self._bars_view):
            if i == self.goal_bar: bar.set_color(0.0, 0.4, 0.0)
            else: bar.set_color(0.3, 0.3, 0.0)

        # move discs to their positions
        disc_placed = 3*[0]
        disc_colors = [
            (0.0, 0.7, 0.7),
            (0.0, 0.0, 1.0),
            (0.7, 0.0, 0.7),
            (1.0, 0.0, 0.0),
            (0.7, 0.7, 0.0),
        ]
        disc_goal_color = (0, 0.7, 0)
        for i,(b,trans,disc) in enumerate(zip(reversed(self.bars), self._discs_view_trans, self._discs_view)):
            brightness = np.interp(i, [0,self.size-1], [0, 0.5])
            if self.size-1-i == self.goal_disc: disc.set_color(*disc_goal_color)
            else: disc.set_color(*disc_colors[i % len(disc_colors)])
            y = screen_height / 2
            y = np.interp(disc_placed[b], [-0.5, self.size-0.5], [y-bar_height/2, y+bar_height/2])
            trans.set_translation(bar_x(b), y)
            disc_placed[b] += 1

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_state(self):
        return {
            'bars' : np.array(self.bars),
            'goal' : np.array((self.goal_disc, self.goal_bar)),
        }
    def goal_accomplished(self):
        return self.bars[self.goal_disc] == self.goal_bar

    def action_i_to_b(self, i):
        return i // 2, ((i + 3) % 6) // 2
    def action_b_to_i(self, b1, b2):
        return b1*2 + (b2-b1-1)%3

    def available_actions_b(self):
        b1 = self.bars[0]
        yield b1, (b1+1)%3
        yield b1, (b1+2)%3
        b2 = next(b for b in self.bars if b != b1)
        yield b2, 3-b1-b2
    def available_actions_i(self):
        return (self.action_b_to_i(b1,b2) for b1,b2 in self.available_actions_b)
    def available_action_mask(self):
        res = np.zeros(6, dtype = bool)
        for i in self.available_actions_i(): res[i] = True
        return res

    def expert_action(self):
        if self.goal_accomplished(): return None

        start_bar = self.bars[self.goal_disc]
        free_bar = 3 - start_bar - self.goal_bar
        for disc in reversed(range(self.goal_disc)):
            bar = self.bars[disc]
            if bar != free_bar:
                start_bar = bar
                free_bar = 3 - bar - free_bar

        goal_bar = 3 - free_bar - start_bar
        return self.action_b_to_i(start_bar, goal_bar)

if __name__ == "__main__":
    from pyglet.window import key
    from pyglet import app, clock

    env = Hanoi()
    env.reset()

    def action_step(action):
        if action is None: return
        print("{} -> {}".format(*env.action_i_to_b(action)))
        print(env.step(action)[2])
        env.render()

    def sol_step(*args):
        action_step(env.expert_action())

    def key_press(k, mod):
        global env
        action_d = {
            key.Z : (1,0),
            key.X : (0,1),
            key.C : (2,1),
            key.V : (1,2),
            key.S : (2,0),
            key.F : (0,2),
        }
        if k in action_d: action_step(env.action_b_to_i(*action_d[k]))
        elif k == key.ENTER:
            sol_step()
            clock.schedule_interval(sol_step, 0.1)
        elif k == key.R:
            env.reset()
            env.render()
        elif k == key.ESCAPE:
            env.close()
            app.exit()

    def key_release(k, mod):
        if k == key.ENTER: clock.unschedule(sol_step)

    def on_draw(*args):
        env.viewer.render()

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.viewer.window.on_expose = on_draw
    env.viewer.window.on_draw = on_draw
    app.run()
