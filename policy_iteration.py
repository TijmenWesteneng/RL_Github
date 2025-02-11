from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np


def policy_iteration(env, gamma, theta):
    action = env.action_space.sample()
    nA = env.action_space.n
    nS = env.observation_space.n
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype = 'int')
    policy += action
    
    while True:
        #Policy evaluation
        while True:
            delta = 0
            for i in range(nS):
                value = V[i]
                a = policy[i]
                v_updated = 0
                #probability, new_state, reward, terminated
                for pob in env.P[i][a]:
                    v_updated += pob[0]*(pob[2] + gamma*V[pob[1]])
                V[i] = v_updated
                delta = max(delta, abs(value - V[i]))
            if delta < theta:
                break
        #Policy improvement        
        policy_stable = True  
        
        for j in range(nS):
            old_ac = policy[j]
            q_value = np.zeros(nA)
            
            for a in range(nA):
                q = 0
                for k in env.P[j][a]:
                    q += k[0]*(k[2] + gamma*V[k[1]])
                q_value[a] = q
                
            policy[j] = np.argmax(q_value)
            if old_ac != policy[j]:
                policy_stable = False
        
        if policy_stable == True:
            break
    return(V, policy)


env = ZombieEscapeEnv(render_mode='ansi', fixed_seed = 42)
env.reset()
V,policy = policy_iteration(env, 0.84, 0.0000000001)

terminal = False
while not terminal:
    action = policy[env.s]
    next_state, reward, terminal = env.fixed_step(action)[:3]
    env.render()
    state = next_state
env.close()
