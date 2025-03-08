from ZombieEscapeEnv import ZombieEscapeEnv
from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
import numpy as np
from monte_carlo_prediction import MonteCarloPrediction
from monte_carlo_control import MonteCarloControl
from TD0 import TD_Prediction
# Create and test the environment
#env = ZombieEscapeEnv(render_mode=None, fixed_seed = 35)
env = ZombieEscapeEnv(render_mode=None, fixed_seed = 35, gamma=0.93)
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
""" #Value iteration as ground truth
value_iteration = ValueIteration(env, 0.00001)
V, policy = value_iteration.get_training_results()
value_iteration.visualise_values()
value_iteration.visualise_policy() """
# print("POLICIES")
policy_iteration = PolicyIteration(env, 0.00001)
V, policy = policy_iteration.get_training_results()
#policy_iteration.visualise_values()
#policy_iteration.visualise_policy()
""" mc_learning = MonteCarloPrediction(zombie_environment=env, policy=policy, episodes=10000, max_steps=100, target_values=V)
V,policy = mc_learning.get_training_results()
mc_learning.plot_error() """
""" mc_control = MonteCarloControl(zombie_environment=env, episodes=100000, max_steps=100, target_values=V)
V,policy = mc_control.get_training_results()
mc_control.visualise_values()
mc_control.visualise_policy()
mc_control.plot_error() """
td_learning = TD_Prediction(zombie_environment=env, policy=policy, alpha=0.9, episodes=10000, target_values=V)
predicted_V, predicted_policy = td_learning.get_training_results()
td_learning.visualise_values()
#td_learning.visualise_policy()
td_learning.plot_error()

# visualise_values(V)

# values = mc_prediction(env, policy, 10000, 0.93)

# visualise_values(values)



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

