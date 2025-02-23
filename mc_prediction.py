from ZombieEscapeEnv import ZombieEscapeEnv
import numpy as np


def mc_prediction(env, policy, num_episodes, gamma):
    """
    Estimates the Value function of a given policy using on-policy, first-vist Monte Carlo prediction.

    Args:
        env: The enviroment
        policy: A policy as np.array that contains an action for each state in the environment
        num_episodes: The number of episodes generated in the monte carlo algorithm that are used to estimate the state values of the policy
        gamma: The discount factor gamma
    
    Returns:
        values: The estimated value function of the policy
    """

    values = np.zeros(env.observation_space.n)
    returns = {state: [] for state in range(len(values))} # Initialize an emtpy list for each state

    for episode_id in range(num_episodes):  # Loop for the number of episodes
        state, _ = env.reset()
        episode = []

        reward = 0
        terminal = False
        while True: # Generate episodes where the actions are performed following the policy until in a terminal state
            action = policy[state]
            episode.append((state, action, reward))
            if terminal:
                break
            next_state, reward, terminal, _, _ = env.step(action)
            
            state = next_state

        G = 0
        for t in range(len(episode) -1, -1, -1): # Loop from last index to first index of the episode
            state, action, reward = episode[t]

            G = gamma * G + reward

            if state not in [x[0] for x in episode[:t]]: # Check if states occurs somewhere earlier in the episode
                returns[state].append(G)
                values[state] = np.mean(returns[state])
    
    return values


