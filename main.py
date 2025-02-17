from ZombieEscapeEnv import ZombieEscapeEnv
from policy_iteration import policy_iteration
from value_iteration import ValueIteration
# Create and test the environment
env = ZombieEscapeEnv(render_mode='human', fixed_seed = 35)
state, info = env.reset()
"""
# Reset environment and render
state = env.reset()
env.render()

env.step(2)

env.render()
"""
'''
terminal = False
while not terminal:
    action = env.action_space.sample()
    next_state, reward, terminal = env.step(action)[:3]
    env.render()
    state = next_state
env.close()
#episode = env.sample_episode()
# print("Collected Episode:", episode)

'''
#V,policy = policy_iteration(env, 0.93, 0.0000000001)
#print(policy)
new_value, new_policy = ValueIteration(env, 0.94, 0.01).get_training_results()
print(new_policy)

terminal = False
while not terminal:
    action = new_policy[env.s]
    print(action)
    next_state, reward, terminal = env.step(action)[:3]
    env.render()
    state = next_state
env.close()

