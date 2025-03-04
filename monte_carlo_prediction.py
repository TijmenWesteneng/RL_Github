from monte_carlo_methods import MonteCarloLearning
from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np

class MonteCarloPrediction(MonteCarloLearning):

    def __init__(self, zombie_environment, policy, episodes = 100, max_steps = 50, target_values=None):
        super().__init__(zombie_environment=zombie_environment, max_steps=max_steps, episodes=episodes)
        self.episodes = episodes
        self.policy = policy
        self.value_function = np.zeros(self.number_of_states)
        self.count_state_visits = np.zeros(self.number_of_states)
        self.state_returns = np.zeros(self.number_of_states)

    
    def run_training(self):
        """
        Process:

        1 - For each episode do.
            1- random initial state selection 
            2 - episode generation based on policy and initial state
            3 - update the count and sum rewards (calling g unction with mode)
            
        2 - compute average
        
        """
        for episode_number in range(self.episodes):
            #Only generate complete episodes for monte carlo methods
            truncated = True
            while truncated:
                # Generate a random non-terminal starting state and random action
                random_state = np.random.randint(0, self.number_of_states)
                while self.zombie_environment.is_terminal(random_state):
                    random_state = np.random.randint(0, self.number_of_states)

                random_action = np.random.randint(0, self.number_of_actions)

                # generate an episode starting at the random starting state and following the policy
                episode, truncated = self.zombie_environment.generate_episode(policy=self.policy, initial_state=random_state, initial_action=random_action, max_steps=self.max_steps)
                
            # update count and reward using calculate_expected_return
            self.calculate_expected_return(episode=episode, gamma=self.gamma, mode="state_value")

            print(f"Episode: {episode_number}")
            # Average the returns to compute the value function
            #self.value_function = np.where(self.count_state_visits > 0, self.state_returns / self.count_state_visits, 0)  # Assign 0 where count_state_visits is 0
            self.value_function = np.divide(
                self.state_returns,
                self.count_state_visits,
                out=self.value_function,
                where=self.count_state_visits>0
            )
            if self.target_values is not None:
                self.store_error(episode_number)

    def store_error(self, episode_number):
        #calculate squared error
        self.errors[episode_number] = np.mean( (self.value_function - self.target_values) ** 2 )

    def plot_error(self):
        import matplotlib.pyplot as plt

        x = list(range(len(self.errors)))
        
        plt.plot(x, self.errors)
        # Labels and title
        plt.xlabel("episodes")
        plt.ylabel("mean squared error")
        plt.legend()
        plt.show()


