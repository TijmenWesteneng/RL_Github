import matplotlib.pyplot as plt
import numpy as np
from ZombieEscapeEnv import ZombieEscapeEnv

class LearningAlgorithm:
    """
    This class represents the base case for a learning algorithm. It contains methods shared by all algorithms such as getters and plotting.
    """
    def __init__(self, zombie_environment:ZombieEscapeEnv):
        #INITIALIZE CLASS VAR
        self.trained = False
        self.value_function = None
        self.policy = None
        #INITIALIZE ENVIRONMENT VALUES
        self.zombie_environment = zombie_environment
        self.number_of_actions = zombie_environment.action_space.n
        self.number_of_states = zombie_environment.observation_space.n
        self.gamma = self.zombie_environment.get_gamma()

        # Initialize list consisting of tuples of episode number and cumulative reward for that episode
        self.cum_reward_list = []

    def initialize_value_function(self):
        """
        Initialize value function as 0 on non terminal states and as reward for terminal states.
        """
        self.value_function = np.zeros(self.number_of_states)
        for state in range(self.number_of_states):
            if self.zombie_environment.is_terminal(state):
                self.value_function[state] = self.zombie_environment.get_state_reward(state)

    def run_training(self):
        pass

    def get_training_results(self):
        #If model has not been trained
        if not self.trained:
            self.run_training()
            self.trained = True
        
        return self.value_function, self.policy
    
    def visualise_policy(self):
        policy_matrix = self.policy.reshape(8,8)

        arrows = {0:(-1,0), 1:(0,-1), 2:(1,0), 3:(0,1)} # For each action, we define the x and y direction of the arrow (e.g., for action 0, i.e. left, the arrow should point in direction (-1, 0))
        scale = 0.3

        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_xticks(np.arange(8) - 0.5)
        ax.set_yticks(np.arange(8) - 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle='--', color='gray', alpha=0.5)

        for r in range(8):
            for c in range(8):
                action = policy_matrix[r, c]
                dx, dy = arrows[action] # Get the arrow x and y directions
                ax.arrow(c, 7 - r, dx * scale, dy * scale, head_width=0.2, head_length=0.2, fc='blue', ec='blue') # create an arrow for the cell 

        ax.set_title("Final policy")
        ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
        plt.show()

    def visualise_values(self, title=""):
        value_matrix = np.flip(self.value_function.reshape(8,8), axis=0)

        ig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_xticks(np.arange(8) - 0.5)
        ax.set_yticks(np.arange(8) - 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle='--', color='gray', alpha=0.5)

        for r in range(8):
            for c in range(8):
                value = value_matrix[r, c].round(2)
                ax.text(c, r, value, ha='center', va='center')

        ax.set_title(title)
        ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
        plt.show()

    def calc_policy_reward(self, episode_n):
        """
        Calculates and saves cumulative reward for current policy.
        TODO: Allow for non-deterministic policies
        """
        cum_reward = 0

        state = self.zombie_environment.reset()[0]
        state_n = 0
        terminal = False
        while not terminal:
            action = self.policy[state]
            next_state, reward, terminal = self.zombie_environment.step(action)[:3]
            # TODO: Check if this cumulative reward calc is correct
            cum_reward += self.gamma ** state_n * reward
            state = next_state

            # To prevent infinite loops, episodes max out at 10000 visited states
            state_n += 1
            if state_n > 10000:
                cum_reward = -100
                break

        self.cum_reward_list.append((episode_n, cum_reward))

    def plot_cum_reward(self):
        index, cum_reward_list = zip(*self.cum_reward_list)
        plt.scatter(index, cum_reward_list)
        plt.xlabel("Episode number")
        plt.ylabel("Cumulative reward")
        plt.title("Cumulative reward of policy over episodes")
        plt.show()