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
        for state in range(self.Q_S_A.shape[0]):
            if self.zombie_environment.get_letter(state) == 'C':
                self.Q_S_A[state,]=-1000
            elif self.zombie_environment.get_letter(state) == 'D':
                self.Q_S_A[state,]=100
        
        
        
    def run_training(self):
        for episode in range(self.episodes):
            state, info = self.zombie_environment.reset()
            terminated =  False
            
            while not terminated:
                action = self.policy[state]
                next_state, _, terminated = self.zombie_environment.step(action)[:3]
                next_action = self.policy[next_state]
                self.Q_S_A[state][action] += self.alpha * (self.zombie_environment.get_state_reward(state) + self.gamma * self.Q_S_A[next_state][next_action] -  self.Q_S_A[state][action])
                state = next_state

            self.value_function = np.max(self.Q_S_A, axis=1)
            if self.target_values is not None:
                self.store_error(episode)

        

    def store_error(self, episode_number):
        #calculate squared error
        self.errors[episode_number] = np.mean( (self.value_function - self.target_values) ** 2 )

    def plot_error(self):
        import matplotlib.pyplot as plt

        x = list(range(len(self.errors)))
        
        plt.plot(x, self.errors)
        # Labels and title
        plt.xlabel("episodes")
        plt.ylabel("mean squared error")
        plt.legend()
        plt.show()            
                
'''
def __init__(self, zombie_environment:ZombieEscapeEnv, alpha, policy, gamma, episodes = 1000):
    super().__init__()
    self.episodes = episodes
    self.zombie_environment = zombie_environment
    self.alpha = alpha
    self.policy = policy
    self.gamma = gamma
    self.value_function = np.zeros(self.zombie_environment.observation_space.n)
    
    
    
def run_training(self):
    for episode in range(self.episodes):
        state, info = self.zombie_environment.reset()
        terminated =  False
        
        while not terminated:
            action = self.policy[state]
            next_state, reward, terminated = self.zombie_environment.step(action)[:3]
            self.value_function[state] += self.alpha*(reward + self.gamma*self.value_function[next_state] - self.value_function[state])
            state = next_state
    '''        
            
            
            
        