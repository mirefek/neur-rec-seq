from pyglet.window import key
from pyglet import app, clock

def run_interactive(env, key_to_action, seed = None):
    if seed is not None: env.seed(seed)
    env.reset()

    if hasattr(env, "action_to_str"):
        action_to_str = env.action_to_str
    else:
        action_to_str = lambda a: "action {}".format(a)

    def action_step(action):
        if action is None: return
        print(action_to_str(action))
        if env.step(action)[2]: print("DONE")
        else: print("... not done yet ...")
        env.render()

    def sol_step(*args):
        action_step(env.expert_action())

    def key_press(k, mod):
        if k in key_to_action: action_step(key_to_action[k])
        elif k == key.ENTER:
            sol_step()
            sol_sleep = 1 / env.metadata.get('video.frames_per_second', 0.5)
            clock.schedule_interval(sol_step, sol_sleep)
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

