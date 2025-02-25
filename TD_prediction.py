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
        
        
        
    def run_training(self):
        for episode in range(self.episodes):
            state, info = self.zombie_environment.reset()
            terminated =  False
            
            while not terminated:
                action = self.policy[state]
                next_state, reward, terminated = self.zombie_environment.step(action)[:3]
                self.value_function[state] += self.alpha*(reward + self.gamma*self.value_function[next_state] - self.value_function[state])
                state = next_state
                
        