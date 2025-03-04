from learning_algorithm import LearningAlgorithm
from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np

class TD_Prediction(LearningAlgorithm):

    def __init__(self, zombie_environment:ZombieEscapeEnv, alpha, policy, gamma, episodes = 1000):
        super().__init__()
        self.episodes = episodes
        self.zombie_environment = zombie_environment
        self.alpha = alpha
        self.policy = policy
        self.gamma = gamma
        self.value_function = np.zeros(self.zombie_environment.observation_space.n)
        for state in range(len(self.value_function)):
            if self.zombie_environment.get_letter(state) == 'C':
                self.value_function[state] = -1000
            elif self.zombie_environment.get_letter(state) == 'D':
                self.value_function[state] = 100
        
        
        
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
    def __init__(self, zombie_environment:ZombieEscapeEnv, alpha, policy, gamma, episodes = 1000):
        super().__init__()
        self.episodes = episodes
        self.zombie_environment = zombie_environment
        self.alpha = alpha
        self.policy = policy
        self.gamma = gamma
        #initial state, action values
        self.Q_S_A = np.zeros((self.zombie_environment.observation_space.n, self.zombie_environment.action_space.n))
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
                next_state, reward, terminated = self.zombie_environment.step(action)[:3]
                next_action = self.policy[next_state]
                self.Q_S_A[state][action] += self.alpha * (reward + self.gamma * self.Q_S_A[next_state][next_action] -  self.Q_S_A[state][action])
                
                
                
'''