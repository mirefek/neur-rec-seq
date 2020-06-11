from gym.envs.registration import register

register(
    id='Recursive-Hanoi-v0',
    entry_point='gym_recursive.envs:Hanoi',
    kwargs={'size' : 8},
    max_episode_steps=200,
)
register(
    id='Recursive-Bottles-v0',
    entry_point='gym_recursive.envs:Bottles',
    kwargs={'size' : 7},
    max_episode_steps=500,
)
