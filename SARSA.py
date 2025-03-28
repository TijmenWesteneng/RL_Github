from ZombieEscapeEnv import ZombieEscapeEnv
from learning_algorithm import LearningAlgorithm
import numpy as np
import random
import matplotlib.pyplot as plt


class SARSA(LearningAlgorithm):
    def __init__(self, zombie_environment:ZombieEscapeEnv, episodes = 1000, alpha = 0.1, target_values=None, decrease_rate=10):
        super().__init__(zombie_environment=zombie_environment)
        self.episodes = episodes
        self.alpha = alpha
        self.zombie_environment = zombie_environment
        self.target_values = target_values
        self.errors = np.zeros(self.episodes)
        self.decrease_rate = decrease_rate
      
        #initial state, action values
        self.Q_S_A = np.zeros((self.number_of_states, self.number_of_actions))
            
            
    
    
    
    def epsilon_policy(self, epsilon, Q_S_A, state, env):
        '''
        Applying epsilon greedy stratgy,
        return the corresponding action given current epsilon
        '''
        prob = random.uniform(0, 1)
        if prob < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_S_A[state])
        return action
   


    def run_training(self):
        for episode in range(self.episodes):
            #restrat the environment after every episode
            state = self.zombie_environment.reset()
            epsilon =  np.exp(-self.decrease_rate/self.episodes*episode) #adjust epsilon to decay gradually                  
            action = self.epsilon_policy(epsilon, self.Q_S_A, state, self.zombie_environment)
            terminated = False
            

            while not terminated:
                next_state, reward, terminated = self.zombie_environment.step(action)[:3]
                next_action = self.epsilon_policy(epsilon, self.Q_S_A, next_state, self.zombie_environment)
                #Apply sarsa state action update function
                self.Q_S_A[state][action] += self.alpha * (reward + self.gamma * self.Q_S_A[next_state][next_action] -  self.Q_S_A[state][action])                    
                #update state action for the next step of episode    
                state, action = next_state, next_action
                
            # Every x episodes calculate the cumulative reward of the current policy
            if episode % 100 == 0:
                self.policy = np.argmax(self.Q_S_A, axis=1)
                self.calc_policy_reward(episode_n = episode)
                 
            #Get the value table after each episode
            self.value_function = np.max(self.Q_S_A, axis = 1)
            
            if self.target_values is not None:
                self.store_error(episode)
                
            self.policy = np.argmax(self.Q_S_A, axis=1)
        return self.Q_S_A, self.policy

    

        
            
