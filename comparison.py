import matplotlib.pyplot as plt
from typing import List

from ZombieEscapeEnv import ZombieEscapeEnv
from learning_algorithm import LearningAlgorithm
from value_iteration import ValueIteration
from td_q_learning import TDQLearning
from SARSA import SARSA

def compare_algs(alg_list: List[LearningAlgorithm]):
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
        plt.plot(index, alg.errors, label=f"{type(alg).__name__}")

    # Labels and visualisation
    plt.xlabel('Episodes')
    plt.ylabel('Root Mean Squared Error')
    plt.title("RMSE values in Q-Learning for different alphas")
    plt.legend(title="Algorithms")
    plt.show()

# Create the environment we'll use to train
env = ZombieEscapeEnv(render_mode=None, fixed_seed=71, gamma=0.93)

# Running policy iteration as a comparison baseline
value_iteration = ValueIteration(env, 0.00001)
vi_v, vi_policy = value_iteration.get_training_results()

td_q_learning = TDQLearning(zombie_environment=env, target_values=vi_v, alpha=0.005)
td_q_learning_2 = TDQLearning(zombie_environment=env, target_values=vi_v, alpha=0.01)

compare_algs([td_q_learning, td_q_learning_2])