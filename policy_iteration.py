from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np
import matplotlib.pyplot as plt


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


def visualise_policy(policy):
    policy_matrix = policy.reshape(8,8)

    arrows = {0:(-1,0), 1:(0,-1), 2:(1,0), 3:(0,1)} # For each action, we define the x and y direction of the arrow (e.g., for action 0, i.e. left, the arrow should point in direction (-1, 0))
    scale = 0.3

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xticks(np.arange(8) - 0.5)
    ax.set_yticks(np.arange(8) - 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)

    for r in range(8):
        for c in range(8):
            action = policy_matrix[r, c]
            dx, dy = arrows[action] # Get the arrow x and y directions
            ax.arrow(c, 7 - r, dx * scale, dy * scale, head_width=0.2, head_length=0.2, fc='blue', ec='blue') # create an arrow for the cell 

    ax.set_title("Final policy")
    ax.set_aspect('equal') # ensure that scaling of x and y is equal so the grid remains square
    plt.show()
