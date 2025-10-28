#===================================================================
  #The Stacking Grid Environment#
#===================================================================

import gymnasium as gym
from .parameters import *
import random as rd
import numpy as np
from itertools import product
import pygame
from collections import deque

#===================================================================

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, grid_size = (NUM_ROWS, NUM_COLS), num_obstacles = NUM_OBSTACLES, fixed_layout = True):
        super(GridEnv, self).__init__() # inherit from gym.Env
        self.num_rows = grid_size[0]
        self.num_cols = grid_size[1]
        self.window_size = 512  # The size of the PyGame window

        # Initialize the 2D grid environment
        self.grid = np.zeros((self.num_rows, self.num_cols))
        self.num_obstacles = num_obstacles
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right

        # Modified observation space to be 4-channel
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_rows, self.num_cols, 4), dtype=np.float32)

        self.terminated = False
        self.truncated = False

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.fixed_layout = fixed_layout
        self.current_layout = None

        self.robot_position = None
        self.obstacle_positions = None
        self.target_position = None

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.steps_taken = 0
        self.max_steps = 50

        if self.fixed_layout:
            # Generate layout once on first reset, then reuse
            if self.current_layout is None:
                self.current_layout = self._generate_layout()
            # Use the stored layout
            self.robot_position, self.obstacle_positions, self.target_position = self.current_layout
        else:
            # Generate a new layout each reset
            self.robot_position, self.obstacle_positions, self.target_position = self._generate_layout()

        self.terminated = False
        self.truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        # Ensure action is valid
        action = int(np.asarray(action).item())
        assert action in [0, 1, 2, 3], f"Invalid action received: {action}"

        self.steps_taken += 1 # Keep track for truncation

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

        # Sparse rewards approach
        reward = -0.1  # Small step penalty

        if 0 <= new_row < self.num_rows and 0 <= new_col < self.num_cols:
            new_position = (new_row, new_col)
            
            if new_position in self.obstacle_positions:
                reward = -10.0 # Penalty for hitting an obstacle
                self.terminated = True
            else:
                self.robot_position = new_position
                
                if self.robot_position == self.target_position:
                    reward = 100.0 # Reward for reaching the target
                    self.terminated = True
                else:
                    self.terminated = False
        else:
            reward = -10.0 # Penalty for stepping out of bounds
            self.terminated = True

        # --- Truncation condition ---
        self.truncated = self.steps_taken >= self.max_steps

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

        # Create 4-channel observation
        obs = np.zeros((self.num_rows, self.num_cols, 4), dtype=np.float32)
        
        # Channel 0: Robot position (binary mask)
        obs[self.robot_position[0], self.robot_position[1], 0] = 1.0
        
        # Channel 1: Obstacles
        for obs_pos in self.obstacle_positions:
            obs[obs_pos[0], obs_pos[1], 1] = 1.0
        
        # Channel 2: Target position
        obs[self.target_position[0], self.target_position[1], 2] = 1.0
        
        # Channel 3: Free space (everywhere except entities)
        obs[:, :, 3] = 1.0  # Start with all 1s
        obs[self.robot_position[0], self.robot_position[1], 3] = 0.0
        for obs_pos in self.obstacle_positions:
            obs[obs_pos[0], obs_pos[1], 3] = 0.0
        obs[self.target_position[0], self.target_position[1], 3] = 0.0
        
        return obs
    
    def _generate_layout(self):
        # Initialize positions for all entities
        # caution : we should not have overlapping positions
        all_positions = list(product(range(self.num_rows), range(self.num_cols)))
        # Sample positions until a valid layout is found
        while True:
            robot_position = rd.choice(all_positions)
            available_positions = [pos for pos in all_positions if pos != robot_position]
            sampled_positions = rd.sample(available_positions, self.num_obstacles + 1)
            # Assign positions for obstacles and target
            obstacle_positions = sampled_positions[:-1]
            target_position = sampled_positions[-1]
            if self._target_is_reachable(robot_position, target_position, obstacle_positions):
                break

        return robot_position, obstacle_positions, target_position
    
    def _randomize_layout(self):
        self.current_layout = self._generate_layout()

    def _target_is_reachable(self, robot_position, target_position, obstacle_positions):
        # Implement a simple BFS to check if target is reachable from robot position
        queue = deque([robot_position])
        visited = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        while queue:
            current = queue.popleft()
            if current == target_position:
                return True
            visited.add(current)

            for d in directions:
                neighbor = (current[0] + d[0], current[1] + d[1])
                if (0 <= neighbor[0] < self.num_rows and
                    0 <= neighbor[1] < self.num_cols and
                    neighbor not in visited and
                    neighbor not in obstacle_positions):
                    queue.append(neighbor)

        return False