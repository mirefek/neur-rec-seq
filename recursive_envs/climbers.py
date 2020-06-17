#!/usr/bin/python3

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from collections import deque
import numpy as np

class Climbers(gym.Env):
    """
    Description:
        Two climbers on the opposite sides of a mountain range
        want to meet each other but they must keep their altitudes
        equal the entire time.
    Observation:
        Type: {
           'terrain' : Box(0, 1, 2*num_peaks+1),
           'climbers' : MultiDiscrete([2*num_peaks+1, 2*num_peaks+1]),
        }
        'terrain' is fixed during a single game describing
            the consecutive heights of peaks / valleys
            with zeroes on both ends
        'climbers' describe the positions of the two climbers.
            climber on a position 2*x is at the valley / peak x
            climber on an odd position is on the appropriate hillside
    Actions:
        Type: Discrete(2)
        0 = LEFT
        1 = RIGHT
        the controlled climber is the one at a peak / valley
    Reward and termination:
        Once the climbers meet, the game terminates with reward 1,
        otherwise, the reward is zero.
    Starting State:
        The mountain range is randomly generated alternating valleys / peaks.
        The climbers start at the opposite sides of the mountain range.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(self, num_peaks = 11):
        self.num_peaks = num_peaks
        self.terrain_size = 2*self.num_peaks+1
        self.num_positions = 2*self.terrain_size-1
        self.action_space = spaces.Discrete(2)
        terrain_low = np.zeros(self.terrain_size, dtype = np.float)
        terrain_high = np.ones(self.terrain_size, dtype = np.float)
        self.observation_space = spaces.Dict({
            'terrain' : spaces.Box(terrain_low, terrain_high, dtype = np.float),
            'climbers' : spaces.MultiDiscrete([self.num_positions, self.num_positions]),
        })
        self.seed()
        self.viewer = None
        self.terrain = np.zeros([self.terrain_size], dtype = float)
        self.climbers = None
        self.start_climbers = np.array([0, self.num_positions-1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, goal_min = 0, goal_limit = None):
        self.terrain[2:-1:2] = self.np_random.permutation(self.num_peaks-1)+1 # valleys
        self.terrain[1::2] = self.np_random.permutation(self.num_peaks) + self.num_peaks # peaks
        self.terrain /= (self.terrain_size-2)
        self.climbers = np.array(self.start_climbers)

        return self.get_state()

    def _move_on_hill(self, direction): # 0 = DOWN, 1 = UP
        assert((self.climbers % 2 == 1).all())
        climber_directions = ((self.climbers // 2 + direction) % 2)*2 - 1
        climbers_on_peaks = self.climbers + climber_directions
        if np.diff(climbers_on_peaks) == 0 or (climbers_on_peaks == self.start_climbers).all():
            self.climbers += climber_directions
        else:
            peaks = self.terrain[climbers_on_peaks//2]
            i = np.argmin(peaks * (2*direction-1))
            self.climbers[i] += climber_directions[i]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        if (self.climbers == self.start_climbers).all(): # start position
            self.climbers += [1, -1]
            self._move_on_hill(1)
        elif self.goal_accomplished():
            if action == 0:
                if self.climbers[0] > 0: self.climbers += [-2,-2]
            else:
                if self.climbers[0] < self.num_positions-1: self.climbers += [2, 2]
        else:
            i = np.argmax(self.climbers % 2 == 0)
            d = 1 - (self.climbers[i] // 2) % 2
            if action == 0: self.climbers[i] -= 1
            else: self.climbers[i] += 1
            self._move_on_hill(d)

        done = self.goal_accomplished()
        reward = float(done)
        return self.get_state(), reward, done, {}

    def render(self, mode='human'):

        screen_width = 700
        screen_height = 300
        h_offset = 30
        v_offset = 30
        climber_radius = 5
        active_climber_radius = 8
        mountain_color = (0.5, 0.5, 0.5)
        climber_color = (0, 0, 0)

        h_scale = (screen_width-2*h_offset) / (self.terrain_size-1)
        v_scale = screen_height-2*v_offset

        def rect_coor(l, r, t, b):
            return [(l, b), (l, t), (r, t), (r, b)]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # side ground
            t,b = screen_height-v_offset, screen_height
            for l,r in (0,h_offset), (screen_width-h_offset, screen_width):
                ground = rendering.FilledPolygon(rect_coor(l,r,v_offset,0))
                ground.set_color(*mountain_color)
                self.viewer.add_geom(ground)

            # add mountains
            self._mountains_view = []
            coor = [
                (0,0), (1,1), (2,0),
            ] + [(x,-v_offset/v_scale) for x in (2,0)]
            for i in range(self.num_peaks):
                mountain = rendering.FilledPolygon(list(coor))
                self._mountains_view.append(mountain)
                transform = rendering.Transform(
                    translation=(h_offset + 2*i*h_scale, v_offset),
                    scale=(h_scale, v_scale),
                )
                mountain.add_attr(transform)
                mountain.set_color(*mountain_color)
                self.viewer.add_geom(mountain)

            # add climbers
            self._climbers_trans = []
            for i in range(2):
                climber = rendering.make_circle(radius = 1, res = 10)
                trans = rendering.Transform()
                self._climbers_trans.append(trans)
                climber.add_attr(trans)
                climber.set_color(*climber_color)
                self.viewer.add_geom(climber)

        if self.climbers is None: return None

        # make mountains correct size
        mountain_bottom = [(x,-v_offset/v_scale) for x in (2,0)]
        for i,m in enumerate(self._mountains_view):
            m.v = list(enumerate(self.terrain[2*i:2*i+3])) + mountain_bottom

        # move climbers
        y = self.terrain[self.climbers[np.argmax(self.climbers % 2 == 0)]//2]
        for i, (x,trans) in enumerate(zip(self.climbers, self._climbers_trans)):
            if x % 2 == 1:
                y0 = self.terrain[(x-1)//2]
                y1 = self.terrain[(x+1)//2]
                if y0 < y1: x += np.interp(y, [y0, y1], [-1, +1])
                else: x += np.interp(-y, [-y0, -y1], [-1, +1])
                trans.set_scale(climber_radius, climber_radius)
            else: trans.set_scale(active_climber_radius, active_climber_radius)
            trans.set_translation(h_offset + x*h_scale/2, v_offset + y*v_scale)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_state(self):
        return {
            'terrain' : np.array(self.terrain),
            'climbers' : np.array(self.climbers),
        }
    def goal_accomplished(self):
        return bool(np.diff(self.climbers) == 0)

    def expert_action(self):
        if self.goal_accomplished(): return None

        search_env = Climbers(num_peaks = self.num_peaks)
        search_env.terrain = np.array(self.terrain)
        start_climbers = tuple(map(int, self.climbers))
        def climbers_next(c, action):
            search_env.climbers = np.array(c)
            state, _, done, _ = search_env.step(action)
            return tuple(map(int,state['climbers'])), done

        q = deque()
        used = set()

        used.add(start_climbers)
        for a in 0,1:
            q.append((climbers_next(start_climbers, a), a))
        while q:
            (climbers, done), first_action = q.popleft()
            if done: return first_action
            if climbers not in used:
                for a in 0,1:
                    q.append((climbers_next(climbers, a), first_action))
                used.add(climbers)

        raise Exception("No solution found")

if __name__ == "__main__":
    from interactive import run_interactive
    from pyglet.window import key
    key_to_action = {
        key.LEFT  : 0,
        key.RIGHT : 1,
    }
    run_interactive(Climbers(), key_to_action)
