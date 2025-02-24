from learning_algorithm import LearningAlgorithm
import numpy as np

class MonteCarloLearning(LearningAlgorithm):

    def __init__(self):
        super().__init__()

        self.state_action_value_function = None #The action value function np array [state, action]
        self.state_action_returns = None #Sum of rewards collected for the state action pair, np array [state, action]
        self.count_state_action_visits = None #Count of visits for the state action pair, np array [state, action]

        self.state_returns = None #Sum of rewards collected for the state action pair, np array [state, action]
        self.count_state_visits = None #Count of visits for the state action pair, np array [state, action]

    def calculate_expected_return(self, episode, mode, gamma):
        """
        Calculate first-visit G_t recursively and update the count of visited + returns 
        
        Parameters:
        episode: a list of tuples containing the rewards.
        mode: state or state-action
        """
        #Improve efficeicncy by first doing forward pass and then check if value needs to be added
        episode_length = len(episode)
        returns = np.zeros(episode_length)
        
        G = 0

        #Backwards
        for t in range(episode_length - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            returns[t] = G

        #Forward pass
        visited_states = set()
        visited_state_actions = set()

        for t, episode_step in enumerate(episode):
            
            state, action, reward = episode_step
            
            if mode == "state_value":
                if state not in visited_states:
                    visited_states.add(state)
                    self.state_returns[state] += returns[t]
                    self.count_state_visits[state] += 1

            elif mode == "state_action_value":
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    self.state_action_returns[state, action] += returns[t]
                    self.count_state_action_visits[state, action] += 1
        





