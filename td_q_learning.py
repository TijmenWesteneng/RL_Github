import numpy as np
import random
random.seed(53)
# Loading bar for during training
from tqdm import tqdm
import matplotlib.pyplot as plt

from ZombieEscapeEnv import ZombieEscapeEnv
from value_iteration import ValueIteration
from learning_algorithm import LearningAlgorithm

class TDQLearning(LearningAlgorithm):
    def __init__(self, zombie_environment: ZombieEscapeEnv,
                 episodes = 100000, alpha = 0.05, epsilon_start = 0.9, epsilon_end = 0.1, target_values = None):
        super().__init__(zombie_environment)
        self.zombie_environment = zombie_environment
        self.episodes = episodes
        self.gamma = zombie_environment.get_gamma()
        self.alpha = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start

        self.qsa = np.zeros((self.zombie_environment.observation_space.n, self.zombie_environment.action_space.n))
        self.policy = np.zeros(self.zombie_environment.observation_space.n)

        self.target_values = target_values
        self.errors = np.zeros(self.episodes)

    @staticmethod
    def epsilon_policy(epsilon, Q_S_A, state, env):
        """Choose random action or best action (according to QSA) based on random chance and epsilon"""
        prob = random.uniform(0, 1)
        if prob < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_S_A[state])
        return action

    def new_epsilon(self, episode_number):
        """Calculate epsilon for certain episode number based on decreasing (liner) epsilon formula"""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon_start - episode_number / self.episodes * (self.epsilon_start - self.epsilon_end))

    def run_training(self):
        for episode in tqdm(range(self.episodes)):
            state = self.zombie_environment.reset()

            terminated = False
            while not terminated:
                # Choose and execute action using epsilon greedy policy
                action = self.epsilon_policy(self.epsilon, self.qsa, state, env=self.zombie_environment)
                next_state, reward, terminated = self.zombie_environment.step(action)[:3]

                # Update q values according to update rule
                self.qsa[state, action] += self.alpha * (reward + self.gamma * np.max(self.qsa[next_state]) - self.qsa[state, action])

                # Update current state to next state
                state = next_state

            # Every x episodes calculate the cumulative reward of the current policy
            if episode % 100 == 0:
                self.policy = np.argmax(self.qsa, axis=1)
                self.calc_policy_reward(episode_n = episode)

            # Calculate the value matrix and compare it target values to caclulate RMS
            self.value_function = np.max(self.qsa, axis=1)
            if self.target_values is not None:
                self.store_error(episode)

            # TODO: Better epsilon function, since it's currently under-performing a static epsilon
            self.new_epsilon(episode)

        # Calculate the final policy matrix and value matrix
        self.policy = np.argmax(self.qsa, axis=1)
        self.value_function = np.max(self.qsa, axis=1)

# Create the environment we'll use to train
env = ZombieEscapeEnv(render_mode='ansi', fixed_seed=24, gamma=0.93)

# Running policy iteration as a comparison baseline
value_iteration = ValueIteration(env, 0.00001)
V, policy = value_iteration.get_training_results()

# Create instance of the learning algorithm and use it to train and display the resulting policy and results
td_q_learning = TDQLearning(zombie_environment=env, target_values=V)

td_q_learning.get_training_results()
td_q_learning.plot_cum_reward()
td_q_learning.visualise_policy()
td_q_learning.visualise_values_heatmap()
td_q_learning.plot_error()
print(td_q_learning.qsa)

"""
# The part below is for averaging the cumulative reward over multiple trainings and save it in a file to plot later
average_cum_reward_list = []

for i in tqdm(range(10)):
    td_q_learning = TDQLearning(zombie_environment=env)
    td_q_learning.get_training_results()
    average_cum_reward_list.append(td_q_learning.cum_reward_list)

with open('avg_cum_reward_list.txt', 'w+') as f:
    f.write(str(average_cum_reward_list))

f.close()
"""

# From here on we run the model and display it on the screen, so this is not for training
test_env = ZombieEscapeEnv(render_mode='human', fixed_seed=24)
state = test_env.reset()

terminal = False
while not terminal:
    action = td_q_learning.policy[state]
    next_state, reward, terminal = test_env.step(action)[:3]
    test_env.render()
    state = next_state

