#===================================================================
  #The Stacking Grid Environment#
#===================================================================

import gymnasium as gym
from .parameters import *
import random as rd
import numpy as np
from itertools import product

#===================================================================

class GridEnv(gym.Env):
    def __init__(self, grid_size = (NUM_ROWS, NUM_COLS), num_obstacles = NUM_OBSTACLES):
        super(GridEnv, self).__init__() # inherit from gym.Env
        self.num_rows = grid_size[0]
        self.num_cols = grid_size[1]

        # Initialize the 2D grid environment
        self.grid = np.zeros((self.num_rows, self.num_cols))
        self.num_obstacles = num_obstacles
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        # The different values in the observation space: (0: empty, 1: robot, 2: obstacle, 3: target)
        self.observation_space = gym.spaces.Box(low=0, high=3, shape=(self.num_rows, self.num_cols), dtype=np.int8)
        self.terminated = False
        self.truncated = False

    def reset(self, seed = None):
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

        info = {
            "robot_position": self.robot_position,
            "target_position": self.target_position,
            "obstacle_positions": self.obstacle_positions
        }

        return self.grid, info


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

        info = {
            "robot_position": self.robot_position,
            "target_position": self.target_position,
            "obstacle_positions": self.obstacle_positions
        }

        return self.grid, reward, self.terminated, self.truncated, info
