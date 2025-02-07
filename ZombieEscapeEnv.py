from enum import Enum
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.toy_text.utils import categorical_sample
from typing import List, Optional

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class ZombieEscapeEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, size=8):
        super(ZombieEscapeEnv, self).__init__()

        self.grid_size = size
        self.r_map = self.generate_random_map()
        self.nrow, self.ncol = nrow, ncol = self.r_map.shape

        nA = 4  # actions
        nS = nrow * ncol  # states

        self.initial_state_distrib = np.array(self.r_map == "S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        # Create tuples inside the dictionary for each cell, to store information like this:
        # (transition probability, next state, reward, terminated)
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}


        # Change the position of the agent to an index
        def to_s(row, col):
            return row * ncol + col

        # Constrain the movement of zombie so it doesn't go out of index
        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        # define the reward and new states after action so it can be filled in self.P
        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = self.r_map[new_row, new_col]
            terminated = new_letter in "CD"
            # decide reward
            if new_letter == "B":
                reward = 1
            elif new_letter == "W":
                reward = -1
            elif new_letter == "D":
                reward = 5
            else:
                reward = 0

            return new_state, reward, terminated

        # Fill in self.P
        probability_table = [0.1, 0.8, 0.1]
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = self.r_map[row, col]
                    # if current state is "C" or "D", game over
                    if letter in "CD":
                        li.append((1.0, s, 0, True))
                    else:
                        # for other state, for each action, the probability of going in the right direction
                        # is 0.8, going in the correct direction's left/right's probability is 0.1
                        for i, b in enumerate([(a - 1) % 4, a, (a + 1) % 4]):
                            li.append((probability_table[i], *update_probability_matrix(row, col, b)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

    def is_valid(self, board: List[List[str]], max_size: int) -> bool:
        # use simple dfs to track there's a valid path from start to Dave's house
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    # exceeding the index
                    if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                        continue
                    # Dave's house
                    if board[r_new][c_new] == "D":
                        return True
                    # To avoid zombie being devoured by the chompers
                    if board[r_new][c_new] != "C":
                        frontier.append((r_new, c_new))
        return False

    def generate_random_map(
            self, size: int = 8, seed: Optional[int] = None
    ) -> np.array:
        """Generates a random valid map (one that has a path from start to Dave's house)

        Args:
            size: size of each side of the grid
            seed: optional seed to ensure the generation of reproducible maps

        Returns:
            A random valid map
        """
        valid = False
        board = []  # initialize to make pyright happy
        p1 = 0.7
        p2 = 0.1
        p3 = 0.1
        p4 = 0.1

        np_random, _ = seeding.np_random(seed)

        while not valid:
            board = np_random.choice(["L", "W", "C", "B"], (size, size), p=[p1, p2, p3, p4])
            board[0][0] = "S"
            board[-1][-1] = "D"
            valid = self.is_valid(board, size)
        return board

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions],
                               self.np_random)  # choose the next move based on the probability table of this action
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        return int(self.s), {"prob": 1}

        # Initialize the state
        # self.state = self.get_state()

    def render(self):
        # Print the board by placing the agent at the specified position"""

        agent_row, agent_col = int(self.s) // 8, int(self.s) % 8
        r_map_copy = self.r_map.copy()
        r_map_copy[agent_row, agent_col] = 'Z'  # Place agent at the specified position
        for row in r_map_copy:
            print("".join(row for row in row))
        print("")

    def is_terminal(self):
        # Check current state to see if it is terminal
        agent_row, agent_col = int(self.s) // 8, int(self.s) % 8
        r_map_copy = self.r_map.copy()
        letter = r_map_copy[agent_row, agent_col]
        if letter in 'CD':
            return True
        return False

    def sample_episode(env, max_steps=50):
        """Sample an episode from the environment using a random policy."""
        episode = []
        state, info = env.reset()
        terminal = False

        while not terminal:
            action = env.action_space.sample()
            next_state, reward, terminal = env.step(action)[:3]
            episode.append((state, action, reward))
            env.render()
            state = next_state

        print("Game over!")
        return episode

    def close(self):
        pass