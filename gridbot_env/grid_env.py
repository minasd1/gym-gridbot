#===================================================================
  #The Stacking Grid Environment#
#===================================================================

import gymnasium as gym
from .parameters import *
import random as rd
import numpy as np
from itertools import product
import pygame

#===================================================================

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, grid_size = (NUM_ROWS, NUM_COLS), num_obstacles = NUM_OBSTACLES):
        super(GridEnv, self).__init__() # inherit from gym.Env
        self.num_rows = grid_size[0]
        self.num_cols = grid_size[1]
        self.window_size = 512  # The size of the PyGame window

        # Initialize the 2D grid environment
        self.grid = np.zeros((self.num_rows, self.num_cols))
        self.num_obstacles = num_obstacles
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        # The different values in the observation space: (0: empty, 1: robot, 2: obstacle, 3: target)
        self.observation_space = gym.spaces.Box(low=0, high=3, shape=(self.num_rows, self.num_cols), dtype=np.int8)
        self.terminated = False
        self.truncated = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        # Initialize positions for all entities
        # caution : we should not have overlapping positions
        self.robot_position = (rd.randint(0, self.num_rows - 1), rd.randint(0, self.num_cols - 1))
        all_positions = list(product(range(self.num_rows), range(self.num_cols)))
        available_positions = [pos for pos in all_positions if pos != self.robot_position]
        sampled_positions = rd.sample(available_positions, self.num_obstacles + 1)

        # Assign positions for obstacles and target
        self.obstacle_positions = sampled_positions[:-1]
        self.target_position = sampled_positions[-1]

        # Place robot, obstacles, and target in the grid
        self.grid[self.robot_position] = 1
        for obs_pos in self.obstacle_positions:
            self.grid[obs_pos] = 2
        self.grid[self.target_position] = 3

        self.terminated = False
        self.truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, info


    def step(self, action):
        # Define movement based on action
        movement = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

        # Calculate new position
        new_row = self.robot_position[0] + movement[action][0]
        new_col = self.robot_position[1] + movement[action][1]

        # Check for boundaries
        if 0 <= new_row < self.num_rows and 0 <= new_col < self.num_cols:
            new_position = (new_row, new_col)
            # Check for obstacles
            if new_position not in self.obstacle_positions:
                self.robot_position = new_position
                # Check if the robot reached the target position where the box is located
                if self.robot_position == self.target_position:
                    reward = 1
                    self.terminated = True
                else:
                    reward = 0
                    self.terminated = False
            else: # Robot collides with an obstacle - episode terminated
                self.terminated = True
                reward = -1 # Penalize the robot
        else:
            # The robot moves outside grid boundaries - episode terminated
            self.terminated = True
            reward = -1

        # Update the grid
        self.grid = np.zeros((self.num_rows, self.num_cols))
        self.grid[self.robot_position] = 1
        for obs_pos in self.obstacle_positions:
            self.grid[obs_pos] = 2
        self.grid[self.target_position] = 3

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info
    
    # A slightly adapted render function from Gymnasium's documentation
    # https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.num_rows
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (self.target_position[1] * pix_square_size,  # x-coordinate (column)
                self.target_position[0] * pix_square_size), # y-coordinate (row)
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (
                (self.robot_position[1] + 0.5) * pix_square_size,
                (self.robot_position[0] + 0.5) * pix_square_size,
            ),
            pix_square_size / 3,
        )

        # Now we draw the obstacles
        for obs in self.obstacle_positions:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    (obs[1] * pix_square_size, obs[0] * pix_square_size),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(self.num_rows + 1):
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
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    #===================================================================
        #Auxilliary Methods#
    #===================================================================

    def _get_info(self):
        return {
            "robot_position": self.robot_position,
            "target_position": self.target_position,
            "obstacle_positions": self.obstacle_positions
        }
    
    def _get_observation(self):
        return np.array(self.grid, dtype=np.int8)
