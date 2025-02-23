from learning_algorithm import LearningAlgorithm

class MonteCarloLearning(LearningAlgorithm):

    def __init__(self):
        super().__init__()

        self.state_action_value_function = None #The action value function np array [state, action]
        self.state_action_reward = None #Sum of rewards collected for the state action pair, np array [state, action]
        self.count_state_action_visits = None #Count of visits for the state action pair, np array [state, action]

        self.state_reward = None #Sum of rewards collected for the state action pair, np array [state, action]
        self.count_state_visits = None #Count of visits for the state action pair, np array [state, action]

    def calculate_expected_return(self, episode, mode):
        """
        Calculate first-visit G_t recursively and update the count of visited + returns 
        
        Parameters:
        episode: a list of tuples containing the rewards.
        mode: state or state-action
        """
        pass




