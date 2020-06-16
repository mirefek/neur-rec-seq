#!/usr/bin/python3

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

class Bottles(gym.Env):
    """
    Description:
        Puzzle strongly inspired by 6 Bottles puzzle by Jean Claude Constantin
        (with 7 bottles by default, the 'size' keyword argument)
        The objective is to get a bottle to a prescribed height.
        Bottles can move only UP and DOWN, platter LEFT and RIGHT, and balls inside
        'L'-s. Contrary to the original puzzle, only the bottle / ball on which
        the platter points can move.
    Observation:
        Type: {
           'bottles' : MultiDiscrete(bottles * [3]),
           'balls' : MultiDiscrete(bottles * [2]),
           'platter' : Discrete(bottles),
           'goal' : MultiDiscrete([bottles, 3]),
        }
        bottles: 0 = top, 1 = middle, 2 = bottom
        balls: 0 = left, 1 = right
        platter: the position of the hole in the platter
        goal: (bottle, goal_position)
    Actions:
        Type: Discrete(4)
        Every action corresponds to putting the top disc from a bar i to a bar j.
        Num   Action
        0     platter or ball LEFT
        1     platter or ball RIGHT
        2     bottle UP
        3     bottle DOWN
        If the move is invalid, no action is made.
    Reward and termination:
        Once the goal is accomplished, the game terminates with reward 1,
        otherwise, the reward is zero.
    Starting State:
        Platter is at the position 0,
        Balls and bottles are on random positions so that no bottle
        is at the position 2.
        The goal is assigned randomly so that it is not accomplished.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }
    def __init__(self, size = 7):
        self.size = size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'bottles' : spaces.MultiDiscrete(size * [3]),
            'balls' : spaces.MultiDiscrete(size * [2]),
            'platter' : spaces.Discrete(size),
            'goal' : spaces.MultiDiscrete([size, 3]),
        })
        self.seed()
        self.viewer = None
        self.platter = None
        self.bottles = None
        self.balls = None
        self.goal_bottle = None
        self.goal_pos = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, goal_min = 0, goal_limit = None):
        if goal_limit is None: goal_limit = self.size
        assert(goal_min >= 0 and goal_limit <= self.size and goal_min < goal_limit)
        self.platter = 0
        bb = self.np_random.randint(3, size = self.size)+1
        self.bottles = bb // 2
        self.balls = bb % 2
        self.goal_bottle = self.np_random.randint(goal_min, goal_limit)
        self.goal_pos = (self.bottles[self.goal_bottle] + 1 + self.np_random.randint(2)) % 3

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        if action in self.available_actions():
            move_bottle = action // 2
            direction = (action % 2) * 2 - 1
            if move_bottle: self.bottles[self.platter] += direction
            elif self.bottles[self.platter] < 2: self.platter += direction
            else: self.balls[self.platter] += direction

        done = self.goal_accomplished()
        reward = float(done)
        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        neck_w = 25
        body_w = 80
        ball_size = 20
        hole_size = 25
        hole_dist = 40
        body_bot = -hole_dist*2 + hole_size/2
        body_top = hole_dist*2
        neck_bot = body_top + 10
        neck_top = 3.7*hole_dist
        lid_size = 35
        bottle_color = 0.3, 0.15, 0
        platter_color = 0.5, 0.5, 0.5
        goal_h = 15
        buff = 5
        bottle_dist = body_w + buff
        holder_top = body_top
        platter_depth = 40
        platter_left = bottle_dist * 1.2
        platter_right = 40

        screen_width = int((self.size+2.5)*bottle_dist)
        screen_height = 400

        def rect_coor(w,h,*args):
            if len(args) == 0:
                l, r, t, b = -w/2, w/2, h/2, -h/2
            else: l, r, t, b = w,h,*args
            return [(l, b), (l, t), (r, t), (r, b)]

        def bottle_x(i):
            return screen_width/2 + (i-(self.size-1)/2)*bottle_dist
        def bottle_y(pos):
            return screen_height/2 - (pos-1)*hole_dist
        def ball_x(i, pos):
            center_dist = (body_w - hole_size)/2
            if pos == 0: res = center_dist
            else: res = bottle_dist - center_dist
            return bottle_x(i) + res

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # construct platter
            self._platter_trans = rendering.Transform()
            def add_platter_rect(*args):
                rect = rendering.FilledPolygon(rect_coor(*args))
                self.viewer.add_geom(rect)
                rect.add_attr(self._platter_trans)
                rect.set_color(*platter_color)

            start_x = (screen_width - bottle_dist*self.size - buff)/2
            end_x = screen_width-start_x
            min_x = start_x - bottle_dist - platter_left
            max_x = end_x + platter_right
            start_y = body_bot+screen_height/2 - buff
            add_platter_rect(min_x, start_x - bottle_dist, start_y + hole_dist, start_y)
            add_platter_rect(min_x, start_x, start_y, start_y - hole_dist)
            add_platter_rect(min_x, max_x, start_y - hole_dist, start_y - hole_dist - platter_depth)
            add_platter_rect(start_x+bottle_dist+buff, max_x, start_y, start_y - hole_dist)
            add_platter_rect(end_x, max_x, start_y + hole_dist, start_y)

            # construct bottles
            self._lids_view = []
            self._bottles_trans = []
            for i in range(self.size):
                bottle_trans = rendering.Transform()
                self._bottles_trans.append(bottle_trans)

                def add_bottle_geom(geom):
                    self.viewer.add_geom(geom)
                    geom.add_attr(bottle_trans)
                def add_poly(nodes):
                    poly = rendering.FilledPolygon(nodes)
                    poly.set_color(*bottle_color)
                    add_bottle_geom(poly)
                def add_rect(*args):
                    add_poly(rect_coor(*args))
                def add_side_rect(y1, y2, sgn):
                    l = body_w/2 * sgn
                    r = (body_w/2-hole_size) * sgn
                    add_rect(l,r,y1,y2)

                add_side_rect(body_bot, -hole_dist-hole_size/2, -1)
                add_side_rect(-hole_dist+hole_size/2, -hole_size/2, -1)
                add_side_rect(hole_size/2, body_top, -1)
                add_side_rect(body_bot, hole_dist - hole_size/2, 1)
                add_side_rect(body_top, hole_dist + hole_size/2, 1)
                add_rect(-body_w/2+hole_size, body_w/2-hole_size, body_top, body_bot)
                add_rect(neck_w/2, -neck_w/2, neck_top, neck_bot)
                add_poly([
                    (-body_w/2, body_top),
                    (-neck_w/2, neck_bot),
                    (neck_w/2, neck_bot),
                    (body_w/2, body_top),
                ])
                lid = rendering.Line((-lid_size/2, neck_top), (lid_size/2, neck_top))
                self._lids_view.append(lid)
                lid.linewidth.stroke = 3
                add_bottle_geom(lid)

                holder_line = rendering.Line((0, holder_top), (0, holder_top - 2*hole_dist))
                holder_line.set_color(1, 1, 1)
                add_bottle_geom(holder_line)
                holder_point = rendering.make_circle(radius = 5, res = 10)
                holder_point.set_color(0.6, 0.6, 0.6)
                holder_point.add_attr(rendering.Transform(
                    translation=(bottle_x(i), bottle_y(2) + holder_top)
                ))
                self.viewer.add_geom(holder_point)

            # construct balls
            self._balls_trans = []
            for i in range(self.size):
                ball = rendering.make_circle(radius = ball_size/2)
                ball.set_color(0, 0, 0.8)
                ball_trans = rendering.Transform()
                ball.add_attr(ball_trans)
                self._balls_trans.append(ball_trans)
                self.viewer.add_geom(ball)

                path_nodes = [
                    (ball_x(i,0), bottle_y(0)),
                    (ball_x(i,0), bottle_y(1)),
                    (ball_x(i,1), bottle_y(1)),
                ]
                ball_path = rendering.PolyLine(path_nodes, False)
                ball_path.set_color(0, 0.5, 1)
                self.viewer.add_geom(ball_path)

            goal_nodes = [
                (lid_size/2, neck_top),
                (body_w/2, neck_top + goal_h / 2),
                (body_w/2, neck_top - goal_h / 2),
            ]
            goal = rendering.FilledPolygon(goal_nodes)
            goal.set_color(0, 0.8, 0)
            self._goal_trans = rendering.Transform(
                translation=(screen_width/2, screen_height/2 - hole_dist)
            )
            goal.add_attr(self._goal_trans)
            self.viewer.add_geom(goal)

        if self.bottles is None: return None

        # set goal
        self._goal_trans.set_translation(
            bottle_x(self.goal_bottle), bottle_y(self.goal_pos))
        for i,lid in enumerate(self._lids_view):
            if i == self.goal_bottle: lid.set_color(0, 0.7, 0)
            else: lid.set_color(0.5, 0.5, 0.5)

        # position bottles
        for i,(pos,trans) in enumerate(zip(self.bottles, self._bottles_trans)):
            trans.set_translation(bottle_x(i), bottle_y(pos))
        # position balls
        for i,(pos,trans) in enumerate(zip(self.balls, self._balls_trans)):
            if pos == 0: y = bottle_y(self.bottles[i]-1)
            else: y = bottle_y(1)
            trans.set_translation(ball_x(i,pos), y)
        # postion platter
        self._platter_trans.set_translation(self.platter * bottle_dist, 0)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_state(self):
        return {
            'bottles' : np.array(self.bottles),
            'balls' : np.array(self.balls),
            'platter' : self.platter,
            'goal' : np.array([self.goal_bottle, self.goal_pos]),
        }
    def goal_accomplished(self):
        return self.bottles[self.goal_bottle] == self.goal_pos

    def available_actions(self):
        if self.bottles[self.platter] < 2:
            if self.platter > 0: yield LEFT # platter LEFT
            if self.platter < self.size-1 and \
               (self.platter == 0 or self.bottles[self.platter-1] == 0):
                yield RIGHT   # platter RIGHT
        else:
            if self.balls[self.platter] == 1: yield LEFT # ball LEFT
            else: yield RIGHT # ball RIGHT

        if self.platter == 0 or self.balls[self.platter-1] == 0:
            if self.balls[self.platter] == 0: min_pos = 1
            else: min_pos = 0
            if self.bottles[self.platter] > min_pos: yield UP # bottle UP
            if self.bottles[self.platter] < 2: yield DOWN # bottle DOWN
    def available_action_mask(self):
        res = np.zeros(4, dtype = bool)
        for i in self.available_actions(): res[i] = True
        return res

    def expert_action(self):
        if self.goal_accomplished(): return None

        goal_bottle = self.goal_bottle
        goal_pos = self.goal_pos
        while True:
            #print("({}) -> {}".format(goal_bottle, goal_pos))
            if goal_pos == 0 and self.balls[goal_bottle] == 0:
                if self.bottles[goal_bottle] == 2: return RIGHT
                else: goal_pos = 2
            if goal_bottle > 0 and self.balls[goal_bottle-1] == 1:
                goal_bottle = goal_bottle-1
                goal_pos = 2
                if self.bottles[goal_bottle] == 2: return LEFT
                continue
            for i in reversed(range(goal_bottle-1)):
                if self.bottles[i] > 0:
                    goal_bottle = i
                    goal_pos = 0
                    break
            else: break

        if self.platter == goal_bottle:
            if goal_pos > self.bottles[goal_bottle]: return DOWN
            else: return UP
        elif self.bottles[self.platter] == 2: return UP
        elif self.platter < goal_bottle: return RIGHT
        else: return LEFT

if __name__ == "__main__":
    from interactive import run_interactive
    from pyglet.window import key
    key_to_action = {
        key.LEFT  : 0,
        key.RIGHT : 1,
        key.UP    : 2,
        key.DOWN  : 3,
    }
    run_interactive(Bottles(), key_to_action)
