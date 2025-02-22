from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np
from learning_algorithm import LearningAlgorithm

class PolicyIteration(LearningAlgorithm):
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
        action = self.zombie_environment.action_space.sample()
        self.policy = np.zeros(self.number_of_states, dtype= 'int')
        self.policy += action
        #INITIALIZE VALUE FUNCTION
        self.initialize_value_function()
        
        
        
    def initialize_value_function(self):
        self.value_function = np.zeros(self.number_of_states)
        for state in range(self.number_of_states):
            if self.zombie_environment.is_terminal(state):
                self.value_function[state] = self.zombie_environment.get_state_reward(state)
                
    def policy_evaluation(self):
        while True:
            delta = 0
            #initialize new array to store evaluated values
            new_value_function = np.zeros(self.number_of_states)
            #iterate for all postion in the grid and update value
            for state in range(self.number_of_states):
                original_action = self.policy[state]
                #Avoid updating the values of terminal states
                if self.zombie_environment.is_terminal(state):
                    new_value_function[state] = self.zombie_environment.get_state_reward(state)
                    continue
                
                #initialize value_update to calculate the sum of V(pi) for 4 directions
                value_update = 0
                #implementing Bellman expectation equation
                for prob in self.zombie_environment.P[state][original_action]:
                    value_update += prob[0]*(self.zombie_environment.get_state_reward(state) + self.gamma*self.value_function[prob[1]])
                
                #store the updated value
                new_value_function[state] = value_update
                delta = max(delta, abs(self.value_function[state] - new_value_function[state]))
            
            self.value_function = new_value_function
            if delta < self.theta:
                break
        return 
    
    def policy_improve(self):
        while True:
            #first perform policy evaluation
            self.policy_evaluation()
            
            policy_stable = True
            
            for state in range(self.number_of_states):
                #avoid updating terminal states
                if self.zombie_environment.is_terminal(state):
                    continue
                
                
                original_action = self.policy[state]
                q_value = np.zeros(self.number_of_actions)
                
                #find the best direction(highest expected reward) in current position
                for action in range(self.number_of_actions):
                    #initialize q to calculate q value for each action
                    q_value_action = 0
                    for prob in self.zombie_environment.P[state][action]:
                        q_value_action += prob[0]*(self.zombie_environment.get_state_reward(state) + self.gamma*self.value_function[prob[1]])
                    #update q_value array, the index of the array corresponds to the current direction
                    q_value[action] = q_value_action
                
                self.policy[state] = np.argmax(q_value)
                #check if the policy array already store the optimal position
                #if not cotinue the loop
                if original_action != self.policy[state]:
                    policy_stable = False
                    
            if policy_stable == True:
                break
        return self.value_function, self.policy
    
    def run_training(self):
        self.policy_improve()
        self.trained = True
        return self.value_function, self.policy
                

