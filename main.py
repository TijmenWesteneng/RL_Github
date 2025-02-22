from ZombieEscapeEnv import ZombieEscapeEnv
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
import numpy as np
from plotting import visualise_policy, visualise_values
from mc_prediction import mc_prediction
# Create and test the environment
#env = ZombieEscapeEnv(render_mode=None, fixed_seed = 35)
env = ZombieEscapeEnv(render_mode='ansi', fixed_seed = 35)
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
# print("POLICIES")
V,policy = PolicyIteration(env, 0.93, 0.00001).get_training_results()
visualise_policy(policy)

values = mc_prediction(env, policy, 1000, 0.93)
visualise_values(V)
visualise_values(values)



# new_value, new_policy = ValueIteration(env, 0.93, 0.00001).get_training_results()
# print(new_policy)
# print(np.where(policy != new_policy))

# print("VALUES")
# print(V)
# print(new_value)

# terminal = False
# while not terminal:
#     action = new_policy[env.s]
#     #print(action)
#     next_state, reward, terminal = env.step(action)[:3]
#     env.render()
#     state = next_state
# env.close()

