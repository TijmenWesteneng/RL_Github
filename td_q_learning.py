import numpy as np
import random
random.seed(53)
# Loading bar for during training
from tqdm import tqdm

from ZombieEscapeEnv import ZombieEscapeEnv
from learning_algorithm import LearningAlgorithm

class TDQLearning(LearningAlgorithm):
    def __init__(self, zombie_environment: ZombieEscapeEnv, episodes = 5000, gamma = 0.85, alpha = 0.1, epsilon = 0.8):
        super().__init__(zombie_environment)
        self.zombie_environment = zombie_environment
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.qsa = np.zeros((self.zombie_environment.observation_space.n, self.zombie_environment.action_space.n))

    @staticmethod
    def epsilon_policy(epsilon, Q_S_A, state, env):
        prob = random.uniform(0, 1)
        if prob < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_S_A[state])
        return action

    def run_training(self):
        for episode in tqdm(range(self.episodes)):
            state = self.zombie_environment.reset()[0]

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
            if episode % 10 == 0:
                self.policy = np.zeros(self.zombie_environment.observation_space.n)
                for i in range(len(self.qsa)):
                    self.policy[i] = np.argmax(self.qsa[i])

                self.calc_policy_reward(episode_n = episode)


# Create the environment we'll use to train
env = ZombieEscapeEnv(render_mode='ansi', fixed_seed=36)

# Create instance of the learning algorithm and use it to train and display the resulting policy and results
td_q_learning = TDQLearning(zombie_environment=env)
td_q_learning.get_training_results()
td_q_learning.plot_cum_reward()
td_q_learning.visualise_policy()
print(td_q_learning.qsa)

# The part below is for averaging the cumulative reward over multiple trainings and save it in a file to plot later
"""
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
test_env = ZombieEscapeEnv(render_mode='human', fixed_seed=36)
state = test_env.reset()[0]

terminal = False
while not terminal:
    action = td_q_learning.policy[state]
    next_state, reward, terminal = test_env.step(action)[:3]
    test_env.render()
    state = next_state

