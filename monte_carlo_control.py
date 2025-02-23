from monte_carlo_methods import MonteCarloLearning
from ZombieEscapeEnv import ZombieEscapeEnv

class MonteCarloControl(MonteCarloLearning):

    def __init__(self, zombie_environment:ZombieEscapeEnv, episodes = 50):
        super().__init__()
        self.episodes = episodes
        self.zombie_environment = zombie_environment
    
    def run_training(self):
        """
        Init:
        random policy for non terminal states
        q(s, a) init in 0
        initialize returns matrix

        Process:

        1 - Choose random start from non terminal states.
        2 - Choose random initial action
        3 - Generate random episode following policy and with inital states
        4 - run the function for calculating returns
        5 - update policy.
        """
        pass