from learning_algorithm import LearningAlgorithm
from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np

class TD_Prediction(LearningAlgorithm):
    def __init__(self, zombie_environment:ZombieEscapeEnv, alpha, policy, episodes = 1000, target_values=None):
        super().__init__(zombie_environment=zombie_environment)
        self.episodes = episodes
        self.alpha = alpha
        self.policy = policy
        self.target_values = target_values
        self.errors = np.zeros(self.episodes)
        #initial state, action values
        self.Q_S_A = np.zeros((self.number_of_states, self.number_of_actions))
        
        
        
        
    def run_training(self):
        for episode in range(self.episodes):
            state, info = self.zombie_environment.reset()
            terminated =  False
            action = self.policy[state]
            
            while not terminated:
                next_state, reward, terminated = self.zombie_environment.step(action)[:3]
                next_action = self.policy[next_state]
                self.Q_S_A[state][action] += self.alpha * (reward + self.gamma * self.Q_S_A[next_state][next_action] -  self.Q_S_A[state][action])
                state, action = next_state, next_action

            self.value_function = np.max(self.Q_S_A, axis=1)
            if self.target_values is not None:
                self.store_error(episode)

        

    def store_error(self, episode_number):
        # Create a mask for non-zero values
        nonzero_mask = self.value_function != 0  

        # Compute squared error only for nonzero positions
        if np.any(nonzero_mask):  # Avoid empty selections
            self.errors[episode_number] = np.mean(
                (self.value_function[nonzero_mask] - self.target_values[nonzero_mask]) ** 2
                )
        else:
            self.errors[episode_number] = 0  # Avoid NaN if all values are zero
        '''
        #calculate squared error
        self.errors[episode_number] = np.mean( (self.value_function - self.target_values) ** 2 )'''

    def plot_error(self):
        import matplotlib.pyplot as plt

        x = list(range(len(self.errors)))
        
        plt.plot(x, self.errors)
        # Labels and title
        plt.xlabel("episodes")
        plt.ylabel("mean squared error")
        plt.legend()
        plt.show()            
                

        