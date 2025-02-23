from monte_carlo_methods import MonteCarloLearning
from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np

class MonteCarloControl(MonteCarloLearning):

    def __init__(self, zombie_environment:ZombieEscapeEnv, episodes = 50, gamma=1):
        super().__init__()
        self.episodes = episodes
        self.zombie_environment = zombie_environment
        self.gamma = gamma
        self.policy = np.zeros(self.zombie_environment.observation_space.n, dtype='int')

        self.state_action_value_function = np.zeros((self.zombie_environment.observation_space.n, self.zombie_environment.action_space.n)) #The action value function np array [state, action]
        self.state_action_returns = np.zeros((self.zombie_environment.observation_space.n, self.zombie_environment.action_space.n)) #Sum of rewards collected for the state action pair, np array [state, action]
        self.count_state_action_visits = np.zeros((self.zombie_environment.observation_space.n, self.zombie_environment.action_space.n), dtype='int') #Count of visits for the state action pair, np array [state, action]

    
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


        for episode_iteration in range(self.episodes):
            #Generate random state
            random_state = np.random.randint(0, self.zombie_environment.observation_space.n)
            while self.zombie_environment.is_terminal(random_state):
                random_state = np.random.randint(0, self.zombie_environment.observation_space.n)
            
            random_action = np.random.randint(0, self.zombie_environment.action_space.n)
            # generate episode with random start
            episode = self.zombie_environment.generate_episode(policy=self.policy, initial_state=random_state, initial_action=random_action)
            # update count and reward using calculate_expected_return
            self.calculate_expected_return(episode=episode, gamma=self.gamma, mode="state_action_value")
        
            np.divide(
                self.state_action_returns, 
                self.count_state_action_visits,
                out=self.state_action_value_function,
                where=self.count_state_action_visits>0
            )

            self.value_function = np.max(self.state_action_value_function, axis=1)
            self.policy = np.argmax(self.state_action_value_function, axis=1)
            print(f"Episode: {episode_iteration}")