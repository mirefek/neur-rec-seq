import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class Test(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        return np.array(self.state), 0, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.track = rendering.Line((0, 200), (screen_width, 200))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def test():
    from time import sleep
    from pyglet.window import key
    from pyglet import app

    env = Test()
    print(env.render())
    env.reset()
    sleep(0.1)
    print("done")

    def key_press(k, mod):
        if k == key.ESCAPE:
            env.close()
            app.exit()

    env.viewer.window.on_key_press = key_press
    app.run()

def test2():
    from time import sleep
    env = Test()
    env.reset()
    print(env.render())
    sleep(2)
    env.close()

if __name__ == "__main__":
    from pyglet.window import key
    from pyglet import app

    env = Test()
    env.reset()
    print(env.render())

    def key_press(k, mod):
        global env
        if k == key.ESCAPE:
            env.close()
            app.exit()

    env.viewer.window.on_key_press = key_press
    app.run()
