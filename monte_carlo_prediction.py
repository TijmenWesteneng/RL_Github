from monte_carlo_methods import MonteCarloLearning
from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np

class MonteCarloPrediction(MonteCarloLearning):

    def __init__(self, zombie_environment:ZombieEscapeEnv, policy, gamma, episodes = 100):
        super().__init__()
        self.episodes = episodes
        self.zombie_environment = zombie_environment
        self.policy = policy
        self.gamma = gamma
        self.value_function = np.zeros(self.zombie_environment.observation_space.n)
        self.count_state_visits = np.zeros(self.zombie_environment.observation_space.n)
        self.state_returns = np.zeros(self.zombie_environment.observation_space.n)
    
    def run_training(self):
        """
        Process:

        1 - For each episode do.
            1- random initiial state selection (check)
            2 - episode generation based on policy and initial state
            3 - update the count and sum rewards (calling g unction with mode)
            
        2 - compute average
        
        """
        for _ in range(self.episodes):
            # episode = generate episode
            episode = self.zombie_environment.generate_episode(policy=self.policy)
            # update count and reward using calculate_expected_return
            self.calculate_expected_return(episode=episode, gamma=self.gamma, mode="state_value")
        
        self.value_function = np.where(self.count_state_visits > 0, self.state_returns / self.count_state_visits, 0)  # Assign 0 where count_state_visits is 0


