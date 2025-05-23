import matplotlib.pyplot as plt
import numpy as np
from ZombieEscapeEnv import ZombieEscapeEnv

class LearningAlgorithm:
    """
    This class represents the base case for a learning algorithm. It contains methods shared by all algorithms such as getters and plotting.
    """
    def __init__(self, zombie_environment:ZombieEscapeEnv, episodes = None, target_values = None):
        #INITIALIZE CLASS VAR
        self.trained = False
        self.value_function = None
        self.policy = None
        #INITIALIZE ENVIRONMENT VALUES
        self.zombie_environment = zombie_environment
        self.number_of_actions = zombie_environment.action_space.n
        self.number_of_states = zombie_environment.observation_space.n
        self.gamma = self.zombie_environment.get_gamma()

        # Target values and error array for RMSE calculations and plotting
        if episodes is not None and target_values is not None:
            self.episodes = episodes
            self.target_values = target_values
            self.errors = np.zeros(episodes)

        # Initialize list consisting of tuples of episode number and cumulative reward for that episode
        self.cum_reward_list = []

    def __repr__(self):
        return type(self).__name__

    def run_training(self):
        pass

    def get_training_results(self):
        #If model has not been trained
        if not self.trained:
            self.run_training()
            self.trained = True
        
        return self.value_function, self.policy
    
    def visualise_policy(self):
        """
        Plots the policy on an 8x8 grid. For each state an arrow is plotted in the direction 
        specified by the policy. For terminal states, the name of the terminal state, e.g. Chomper,
        is plotted instead of an arrow.
        """
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
                state_id = r * 8 + c
                # Check if the state is terminal, if yes plot an arrow, if no plot a T
                if not self.zombie_environment.is_terminal(state_id):
                    action = policy_matrix[r, c]
                    dx, dy = arrows[action] # Get the arrow x and y directions
                    ax.arrow(c, 7 - r, dx * scale, dy * scale, head_width=0.2, head_length=0.2, fc='blue', ec='blue') # create an arrow for the cell 
                elif self.zombie_environment.get_letter(state_id) == "C":
                    ax.text(c, 7 - r, "Chomper", fontsize=10, ha='center', va='center')
                elif self.zombie_environment.get_letter(state_id) == "D":
                    ax.text(c, 7 - r, "Dave's\nhouse", fontsize=10, ha='center', va='center')

        ax.set_title('Policy')
        ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
        plt.show()

    def visualise_values(self, title='Value Function', value_function=None):
        """
        Plots the value matrix on an 8x8 grid.
        """
        if value_function is None:
            value_function = self.value_function

        value_matrix = np.flip(value_function.reshape(8,8), axis=0) # Flip the rows so row 0 is at the bottom and row 7 is at the top to be in line with the pygame environment

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
                value = value_matrix[r, c].round(2)
                ax.text(c, r, value, ha='center', va='center')

        ax.set_title(title)
        ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
        plt.show()

    def visualise_values_heatmap(self, title='Value Function', value_function=None):
        """
        Plot the value matrix on an 8x8 grid as a heatmap.
        """
        if value_function is None:
            value_function = self.value_function

        value_matrix = np.flip(value_function.reshape(8,8), axis=0)
        
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(value_matrix, origin='lower')

        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_xticks(np.arange(8) - 0.5)
        ax.set_yticks(np.arange(8) - 0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle='--', color='k', alpha=0.7)

        for r in range(8):
            for c in range(8):
                value = value_matrix[r, c].round(2)
                ax.text(c, r, value, ha='center', va='center', color='w')

        ax.set_title(title)
        ax.set_aspect('equal')
        plt.show()

    def visualise_values_difference(self, values_comparison = None, abs = False, heatmap = False):
        """
        Visualise difference in value function matrices on an 8x8 grid.
        Args:
            values_comparison (np.array): The value function to compare the own value function to
            abs (bool): If the difference should be represented as absolute values (empty: False)
            heatmap (bool): If the difference should be plotted as a heatmap (empty: False)
        """
        if values_comparison is None:
            assert self.target_values is not None, "visualise_values_difference: target_values must be set when no values_comparison is given"
            values_comparison = self.target_values

        subtraction_matrix = np.subtract(self.value_function, values_comparison)

        if abs:
            subtraction_matrix = np.abs(subtraction_matrix)

        if heatmap:
            self.visualise_values_heatmap(title="Value Function Difference", value_function=subtraction_matrix)
        else:
            self.visualise_values(title='Value Function Difference', value_function=subtraction_matrix)

    def calc_policy_reward(self, episode_n):
        """
        Calculates and saves cumulative reward for current policy.
        TODO: Allow for non-deterministic policies
        """
        cum_reward = 0

        state = self.zombie_environment.reset()
        state_n = 0
        terminal = False
        while not terminal:
            action = self.policy[state]
            next_state, reward, terminal = self.zombie_environment.step(action)[:3]
            # TODO: Check if this cumulative reward calc is correct
            cum_reward += self.gamma ** state_n * reward
            state = next_state

            # To prevent infinite loops, episodes max out at 1000 visited states
            state_n += 1
            if state_n > 1000:
                cum_reward = -100
                break

        self.cum_reward_list.append((episode_n, cum_reward))

    def store_error(self, episode_number):
        """Calculate Mean Squared Error and save to error array"""
        self.errors[episode_number] = np.sqrt(np.mean((self.value_function - self.target_values) ** 2))

    def plot_error(self):
        """Plot Mean Squared Error over episodes"""
        x = list(range(len(self.errors)))

        plt.plot(x, self.errors)

        # Labels and title
        plt.xlabel("Episodes")
        plt.ylabel("Root Mean Squared Error")

        plt.show()

    def plot_cum_reward(self):
        if len(self.cum_reward_list) > 0:
            index, cum_reward_list = zip(*self.cum_reward_list)
            plt.scatter(index, cum_reward_list)
            plt.xlabel("Episode number")
            plt.ylabel("Cumulative reward")
            plt.title("Cumulative reward of policy over episodes")
            plt.show()