from ZombieEscapeEnv import ZombieEscapeEnv

# Create and test the environment
env = ZombieEscapeEnv()

"""
# Reset environment and render
state = env.reset()
env.render()

env.step(2)

env.render()
"""

episode = env.sample_episode()
# print("Collected Episode:", episode)