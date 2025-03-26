from monte_carlo_methods import MonteCarloLearning
from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np
from tqdm import tqdm

class MonteCarloControl(MonteCarloLearning):

    def __init__(self, zombie_environment, episodes = 100, max_steps = 50, target_values=None):
        super().__init__(zombie_environment = zombie_environment, max_steps=max_steps, target_values=target_values, episodes=episodes)
        self.episodes = episodes
        self.policy = np.zeros(self.number_of_states, dtype='int')

        self.state_action_value_function = np.zeros((self.number_of_states, self.number_of_actions)) #The action value function np array [state, action]
        self.state_action_returns = np.zeros((self.number_of_states, self.number_of_actions)) #Sum of rewards collected for the state action pair, np array [state, action]
        self.count_state_action_visits = np.zeros((self.number_of_states, self.number_of_actions), dtype='int') #Count of visits for the state action pair, np array [state, action]

    
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


        for episode_number in tqdm(range(self.episodes)):
            #Only generate complete episodes for monte carlo methods, avoid them being too long
            truncated = True
            while truncated:
                #Generate random state that is not terminal
                random_state = np.random.randint(0, self.number_of_states)
                while self.zombie_environment.is_terminal(random_state):
                    random_state = np.random.randint(0, self.number_of_states)
                
                random_action = np.random.randint(0, self.number_of_actions)
                # generate episode with random start
                episode, truncated = self.zombie_environment.generate_episode(policy=self.policy, initial_state=random_state, initial_action=random_action, max_steps=self.max_steps)
                # update count and reward using calculate_expected_return
            
            self.calculate_expected_return(episode=episode, gamma=self.gamma, mode="state_action_value")
            
            self.state_action_value_function = np.divide(
                self.state_action_returns, 
                self.count_state_action_visits,
                out=self.state_action_value_function,
                where=self.count_state_action_visits>0
            )

            self.value_function = np.max(self.state_action_value_function, axis=1)
            self.policy = np.argmax(self.state_action_value_function, axis=1)
            
            if self.target_values is not None:
                self.store_error(episode_number)