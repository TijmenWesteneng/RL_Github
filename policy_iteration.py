from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np


def policy_iteration(env, gamma, theta):
    action = env.action_space.sample()
    nA = env.action_space.n
    nS = env.observation_space.n
    #initialize the V array to store the expected reward in one position
    V = np.zeros(nS)
    #initialize the policy array to store the "best" direction to go for each position, 
    #first initial as a random direction
    policy = np.zeros(nS, dtype = 'int')
    policy += action
    
    while True:
        #Policy evaluation
        while True:
            delta = 0
            #iterate for all postion in the grid and update value
            for cell_postition in range(nS):
                value = V[cell_postition] #original value before evaluation
                old_action = policy[cell_postition]  #original action before evaluation
                v_updated = 0 #initialize v_updated to calculate the sum of V(pi) for 4 directions
                #implementing Bellman expectation equation
                for prob in env.P[cell_postition][old_action]:
                    v_updated += prob[0]*(prob[2] + gamma*V[prob[1]])
                #update the V array value
                V[cell_postition] = v_updated
                delta = max(delta, abs(value - V[cell_postition]))
            if delta < theta:
                break
        #Policy improvement        
        policy_stable = True  
        
        for cell_postition_ in range(nS):
            old_ac = policy[cell_postition_]
            q_value = np.zeros(nA)
            #find the best direction(highest expected reward) in current position
            for actions in range(nA):
                #initialize q to calculate q value for each direction
                q = 0
                for prob_ in env.P[cell_postition_][actions]:
                    q += prob_[0]*(prob_[2] + gamma*V[prob_[1]])
                #update q_value array, the index of the array corresponds to the current direction
                q_value[actions] = q
            
            #find the current optimal action
            policy[cell_postition_] = np.argmax(q_value)
            #check if the policy array already store the optimal position
            #if not cotinue the loop
            if old_ac != policy[cell_postition_]:
                policy_stable = False
        
        if policy_stable == True:
            break
    return(V, policy)



