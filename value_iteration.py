from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np
from learning_algorithm import LearningAlgorithm

class ValueIteration(LearningAlgorithm):
    def __init__(self, zombie_environment:ZombieEscapeEnv, gamma, theta):
        super().__init__()
        #INITIALIZE CONFIG PARAMETERS
        self.gamma = gamma
        self.theta = theta
        #INITIALIZE ENVIRONMENT VALUES
        self.zombie_environment = zombie_environment
        self.number_of_actions = zombie_environment.action_space.n
        self.number_of_states = zombie_environment.observation_space.n
        #INITIALIZE POLICY PROPERTY
        self.policy = np.zeros(self.number_of_states, dtype= 'int')
        #INITIALIZE VALUE FUNCTION
        self.initialize_value_function()
        

    def initialize_value_function(self):
        self.value_function = np.zeros(self.number_of_states)
        for state in range(self.number_of_states):
            if self.zombie_environment.is_terminal(state):
                self.value_function[state] = self.zombie_environment.get_state_reward(state)
        
    
    def single_value_iteration(self):
        #Initialize q values function to store the values for computing max
        new_value_function = np.zeros(self.number_of_states)
        delta = 0
        for state in range(self.number_of_states):                
            #Avoid updating the values of terminal states
            if self.zombie_environment.is_terminal(state):
                new_value_function[state] = self.zombie_environment.get_state_reward(state)
                continue
            
            values = np.zeros(self.number_of_actions) #Store the values of the different actions
            for action in range(self.number_of_actions):
                value = 0
                # prob is a tuple of (transition probability, next state, reward)
                for prob in self.zombie_environment.P[state][action]:
                    value += prob[0]*(self.zombie_environment.get_state_reward(state) + self.gamma*self.value_function[prob[1]]) #The value for an action
                
                values[action] = value
            #Get max value, best action and store the value and best action
            max_value = np.max(values, axis=0)
            best_action = np.argmax(values)
            self.policy[state] = best_action
            new_value_function[state] = max_value
            #update delta
            delta = max(delta, abs(self.value_function[state] - new_value_function[state]))

        #update policy
        self.value_function = new_value_function
        
        return delta
                
                
    def run_training(self):
        #Run value iteration until convergence
        delta = self.single_value_iteration()
        while delta > self.gamma:
            delta = self.single_value_iteration()

        #Given states find optimal policy
        self.trained = True

    


        
        

