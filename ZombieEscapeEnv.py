import numpy as np
import random
import pygame
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

    def __init__(self, render_mode=None, size=8, fixed_seed: Optional[int] = None):
        super(ZombieEscapeEnv, self).__init__()
        #store grid size
        self.grid_size = size
        self.fixed_seed = fixed_seed
        #PYGAME INITIALIZATIONS
        #only load this if needed
        if render_mode == 'human':
            self.window_size = 512  # The size of the PyGame window
            self.brain_img = pygame.image.load("./img/brain.png")  # Brain
            self.grass_img = pygame.image.load("img/grass.jpeg")  # grass
            self.spikeweed_img = pygame.image.load("./img/spikeweed.png")  # spikeweed
            self.plant_img = pygame.image.load("./img/plant.png")  # plant
            self.zombie_img = pygame.image.load("./img/zombie.png")  # zombie
            self.house_img = pygame.image.load("./img/house.png")  # house

            #Resize images to fit grid size
            self.cell_size = self.window_size // self.grid_size
            self.brain_img = pygame.transform.scale(self.brain_img, (self.cell_size, self.cell_size))
            self.grass_img = pygame.transform.scale(self.grass_img, (self.cell_size, self.cell_size))
            self.spikeweed_img = pygame.transform.scale(self.spikeweed_img, (self.cell_size, self.cell_size))
            self.plant_img = pygame.transform.scale(self.plant_img, (self.cell_size, self.cell_size))
            self.zombie_img = pygame.transform.scale(self.zombie_img, (self.cell_size, self.cell_size))
            self.house_img = pygame.transform.scale(self.house_img, (self.cell_size, self.cell_size))
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        #ACTUAL GAME INITIALIZATIONS
        self.r_map = self.generate_random_map(seed=self.fixed_seed)
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
            reward = self.get_reward(new_letter)
            
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

        np_random, _ = seeding.np_random(seed if seed is not None else self.fixed_seed)

        while not valid:
            board = np_random.choice(["L", "W", "C", "B"], (size, size), p=[p1, p2, p3, p4])
            board[0][0] = "S"
            board[-1][-1] = "D"
            valid = self.is_valid(board, size)
        return board

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)  # choose the next move based on the probability table of this action
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p}
    
    def fixed_step(self, a):
        transitions = self.P[self.s][a]        
        p, s, r, t = transitions[1]
        self.s = s
        self.lastaction = a

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p}
    
    def _get_agent_location(self):
        #convert state to row, column
        return self.convert_state_to_table_index(self.s)
    
    def convert_state_to_table_index(self, state):
        return state // self.grid_size, state % self.grid_size
    
    def get_reward(self, letter):
        # decide reward
        reward:float
        if letter == "B":
            reward = 0.5
        elif letter == "W":
            reward = -0.5
        elif letter == "D":
            reward = 100
        elif letter == "C":
            reward = -1000  
        else:
            reward = 0
        
        return reward
    
    def get_letter(self, state):
        row, column =  self.convert_state_to_table_index(state)
        return self.r_map[row, column]
    
    def get_state_reward(self, state):
        letter = self.get_letter(state)
        return self.get_reward(letter)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        super().reset(seed = seed if seed is not None else self.fixed_seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self._render_frame()

        return int(self.s), {"prob": 1}

        # Initialize the state
        # self.state = self.get_state()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.grid_size
        )  # The size of a single grid square in pixels

        # First we draw the house
        target_x, target_y = (self.grid_size - 1, self.grid_size - 1)
        canvas.blit(self.house_img, (target_x * pix_square_size, target_y * pix_square_size))

        #Draw the rest of the objects
        for row in range(self.r_map.shape[0]):
            for col in range(self.r_map.shape[1]):
                letter = self.r_map[col, row]
                match letter:
                    case 'B':
                        canvas.blit(self.grass_img, (row * pix_square_size, col * pix_square_size))
                        canvas.blit(self.brain_img, (row * pix_square_size, col * pix_square_size))
                    case 'L':
                        canvas.blit(self.grass_img, (row * pix_square_size, col * pix_square_size))
                    case 'C':
                        canvas.blit(self.grass_img, (row * pix_square_size, col * pix_square_size))
                        canvas.blit(self.plant_img, (row * pix_square_size, col * pix_square_size))
                    case 'W':
                        canvas.blit(self.grass_img, (row * pix_square_size, col * pix_square_size))
                        canvas.blit(self.spikeweed_img, (row * pix_square_size, col * pix_square_size))
       
        # Now we draw the agent
        agent_y, agent_x  = self._get_agent_location()
        canvas.blit(self.zombie_img, (agent_x * pix_square_size, agent_y * pix_square_size))

        # Finally, add some gridlines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def render(self):
        # Print the board by placing the agent at the specified position"""
        if self.render_mode == 'human':
            return self._render_frame()
        
        if self.render_mode == 'ansi':
            agent_row, agent_col = int(self.s) // self.grid_size, int(self.s) % self.grid_size
            r_map_copy = self.r_map.copy()
            r_map_copy[agent_row, agent_col] = 'Z'  # Place agent at the specified position
            for row in r_map_copy:
                print("".join(row for row in row))
            print("")

    def is_terminal(self, state=None):
        if state == None:
          state = self.s

        # Check current state to see if it is terminal
        agent_row, agent_col = int(state) // self.grid_size, int(state) % self.grid_size
        r_map_copy = self.r_map.copy()
        letter = r_map_copy[agent_row, agent_col]
        if letter in 'CD':
            return True
        return False

    def sample_episode(self, env, max_steps=50):
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
        self.close()
        print("Game over!")
        return episode


    def generate_episode(self, policy, initial_state=0):
        """
        Sample an episode from the env following a policy. (Start at state 0 unless the intial state is specified)
        """

        self.s = initial_state
        state = initial_state
        episode = []

        reward = 0
        terminal = False
        while True: # Generate episodes where the actions are performed following the policy until in a terminal state
            action = policy[state]
            episode.append((state, action, reward))
            if terminal:
                break
            next_state, reward, terminal, _, _ = self.step(action)
            state = next_state
        return episode

    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None  # Reset clock too
