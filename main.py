from ZombieEscapeEnv import ZombieEscapeEnv

# Create and test the environment
env = ZombieEscapeEnv(render_mode='human')
state, info = env.reset()
"""
# Reset environment and render
state = env.reset()
env.render()

env.step(2)

env.render()
"""
terminal = False
while not terminal:
    action = env.action_space.sample()
    next_state, reward, terminal = env.step(action)[:3]
    env.render()
    state = next_state
env.close()
#episode = env.sample_episode()
# print("Collected Episode:", episode)