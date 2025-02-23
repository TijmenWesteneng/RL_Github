from learning_algorithm import LearningAlgorithm

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
        G = 0
        for t in range(len(episode) -1, -1, -1): # Loop from last index to first index of the episode
            state, action, reward = episode[t]

            G = gamma * G + reward

            if state not in [x[0] for x in episode[:t]]:
                
                if mode == "state_value":
                    self.state_returns[state] += G
                    self.count_state_visits[state] += 1

                elif mode == "state_action_value":
                    self.state_action_returns[state, action] += G
                    self.count_state_action_visits[state, action] += 1





