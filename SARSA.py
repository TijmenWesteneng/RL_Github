from ZombieEscapeEnv import ZombieEscapeEnv
from learning_algorithm import LearningAlgorithm
import numpy as np
import random


class SARSA(LearningAlgorithm):
    def __init__(self, zombie_environment:ZombieEscapeEnv, episodes = 1000, gamma = 1, alpha = 0.1):
        super().__init__()
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.zombie_environment = zombie_environment
      
        #initial state, action values
        self.Q_S_A = np.zeros((self.zombie_environment.observation_space.n, self.zombie_environment.action_space.n))
        for state in range(self.Q_S_A.shape[0]):
            if self.zombie_environment.get_letter(state) == 'C':
                self.Q_S_A[state,]=-1000
            elif self.zombie_environment.get_letter(state) == 'D':
                self.Q_S_A[state,]=100
            
            
    
    
    
    def epsilon_policy(self, epsilon, Q_S_A, state, env):
        prob = random.uniform(0, 1)
        if prob < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_S_A[state])
        return action
   


    def run_training(self):
        for episode in range(self.episodes):
            state, info = self.zombie_environment.reset()
            epsilon = max(0.01, 1 / (1 + episode / 100))
            
            action = self.epsilon_policy(epsilon, self.Q_S_A, state, self.zombie_environment)
            terminated = False
           

            
            while not terminated:
                next_state, reward, terminated = self.zombie_environment.step(action)[:3]
                next_action = self.epsilon_policy(epsilon, self.Q_S_A, next_state, self.zombie_environment)
                self.Q_S_A[state][action] += self.alpha * (reward + self.gamma * self.Q_S_A[next_state][next_action] -  self.Q_S_A[state][action])
                

                state, action = next_state, next_action
        self.policy = np.argmax(self.Q_S_A, axis=1)
        self.value_function = np.max(self.Q_S_A, axis = 1)
        return self.Q_S_A, self.policy
     
        
            
env = ZombieEscapeEnv(render_mode="ansi", fixed_seed = 36)
state, info = env.reset() 
env.render()

from policy_iteration import PolicyIteration
policy_iteration = PolicyIteration(env, 0.93, 0.00001)
V, policy = policy_iteration.get_training_results()
print(V)
print(policy)


sarsa = SARSA(env, episodes = 1000, gamma = 0.93)
#print(sarsa.Q_S_A)
V1,policy1 = sarsa.get_training_results()   
print(V1)
print(policy1)

#policy2 = np.array([0, 0, 0, 2, 2, 0, 0, 0, 1, 0, 1, 2, 2, 0, 1, 1, 1, 1, 2, 2, 3, 3, 0, 1, 2, 0, 0, 3, 3, 0, 3, 0, 3, 1, 0, 0, 0,
 #          2, 3, 3, 3, 0, 1, 1, 1, 3, 3, 3, 2, 1, 0, 1, 0, 0, 0, 2, 1, 0, 3, 0, 0, 1, 1, 0])
'''
terminal = False
while not terminal:
    action = policy2[env.s]
    print(action)
    next_state, reward, terminal = env.step(action)[:3]
    env.render()
    state = next_state
env.close()'''