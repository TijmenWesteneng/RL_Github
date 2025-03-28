import matplotlib.pyplot as plt
from typing import List
import numpy as np

from ZombieEscapeEnv import ZombieEscapeEnv
from learning_algorithm import LearningAlgorithm
from value_iteration import ValueIteration
from td_q_learning import TDQLearning
from monte_carlo_control import MonteCarloControl
from SARSA import SARSA

def compare_algs_error(alg_list: List[LearningAlgorithm]):
    # Check if all algs have same amount of episodes
    episode_amount = alg_list[0].episodes
    for alg in alg_list[1::]:
        assert alg.episodes == episode_amount, "Episode amount should be equal for all algorithms to plot difference"

    # Train all algs
    for alg in alg_list:
        print(f"Now training: {alg}")
        alg.get_training_results()

    # Plot errors for all algs
    index = range(len(alg_list[0].errors))
    for alg in alg_list:
        plt.plot(index, alg.errors, label=alg)

    # Labels and visualisation
    plt.xlabel('Episodes')
    plt.ylabel('Root Mean Squared Error')
    plt.title("RMSE values over episodes for different algorithms")
    plt.legend(title="Algorithms")
    plt.show()

def compare_algs_cum(alg_list: List[LearningAlgorithm], training_amount = 2):
    average_cum_reward_list = []
    for j in range(len(alg_list)):
        average_cum_reward_list.append([])

    for i in range(training_amount):
        for j, alg in enumerate(alg_list):
            print(f"Now training: {alg} for iteration {i+1}")
            alg.get_training_results()
            average_cum_reward_list[j].append(alg.cum_reward_list)
            alg.trained = False

    matrix = []
    for i in range(len(alg_list)):
        matrix.append(np.zeros((len(average_cum_reward_list[i]), len(average_cum_reward_list[i][0]))))

        for j in range(len(average_cum_reward_list)):
            index, values = zip(*average_cum_reward_list[i][j])
            matrix[i][j] = np.asarray(values, dtype=np.float32)

        average_array = np.array(matrix[i]).mean(axis=0)
        plt.scatter(index, average_array, label=alg_list[i])
    plt.title(f"Cumulative reward over {training_amount} training iterations")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend(title="Algorithms")
    plt.show()

if __name__ == "__main__":
    # Create the environment we'll use to train
    env = ZombieEscapeEnv(render_mode=None, fixed_seed=71, gamma=0.93)

    # Running policy iteration as a comparison baseline
    value_iteration = ValueIteration(env, 0.00001)
    vi_v, vi_policy = value_iteration.get_training_results()

    td_q_learning = TDQLearning(zombie_environment=env, episodes=100000, target_values=vi_v, alpha=0.01)
    monte_carlo = MonteCarloControl(zombie_environment=env, episodes=100000, target_values=vi_v)

    #compare_algs_error([td_q_learning, monte_carlo])
    compare_algs_cum([td_q_learning, monte_carlo])