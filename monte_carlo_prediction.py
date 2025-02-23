from monte_carlo_methods import MonteCarloLearning
from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np

class MonteCarloControl(MonteCarloLearning):

    def __init__(self, zombie_environment:ZombieEscapeEnv, policy, gamma, episodes = 50):
        super().__init__()
        self.episodes = episodes
        self.zombie_environment = zombie_environment
        self.policy = policy
        self.gamma = gamma
        self.value_function = np.zeros(self.zombie_environment.observation_space.n)
        self.count_state_visits = np.zeros(self.zombie_environment.observation_space.n)
        self.state_reward = np.zeros(self.zombie_environment.observation_space.n)
    
    def run_training(self):
        """
        Process:

        1 - For each episode do.
            1- random initiial state selection (check)
            2 - episode generation based on policy and initial state
            3 - update the count and sum rewards (calling g unction with mode)
            
        2 - compute average
        
        """
        pass