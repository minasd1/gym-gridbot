#===================================================================
  # Test Suite for GridBot Environment #
#===================================================================

import numpy as np
from gridbot_env.grid_env import GridEnv

#===================================================================

# Ensure correct type and content of reset info
def test_reset_returns_correct_info():
    env = GridEnv()
    observation, info = env.reset(seed=42)
    assert isinstance(info, dict)
    assert "robot_position" in info
    assert "target_position" in info
    assert "obstacle_positions" in info
    assert len(info["obstacle_positions"]) == env.num_obstacles

# Ensure step updates robot position correctly
def test_step_updates_robot_position():
    env = GridEnv()
    env.reset(seed=42)
    initial_position = env.robot_position
    observation, reward, terminated, truncated, info = env.step(3)  # Move Right
    new_position = env.robot_position
    assert ( # Check if position is updated or if movement was blocked by boundary/obstacle
        new_position != initial_position
        or initial_position[1] == env.num_cols - 1
        or (initial_position[0], initial_position[1] + 1) in env.obstacle_positions
    )

# Ensure correct type and content of step info
def test_step_returns_correct_info():
    env = GridEnv()
    env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(0)  # Move Up
    assert isinstance(info, dict)
    assert "robot_position" in info
    assert "target_position" in info
    assert "obstacle_positions" in info

# Test robot reaching target position - the reward should be 1 and the agent should reach a terminal state
def test_robot_reaches_target():
    env = GridEnv()
    env.reset(seed=42)

    # Place robot near target (just above, for example)
    target_x, target_y = env.target_position
    if target_x > 0:  # if not on top row
        env.robot_position = (target_x - 1, target_y)
        action = 1  # Move down into target
    else:
        env.robot_position = (target_x + 1, target_y)
        action = 0  # Move up into target

    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 100
    assert terminated is True

# Test robot hitting an obstacle - the reward should be -1 and the agent should reach a terminal state
def test_robot_hits_obstacle():
    env = GridEnv()
    env.reset(seed=42)

    if env.obstacle_positions:
        obstacle_x, obstacle_y = env.obstacle_positions[0]

        # Place robot near obstacle (just above it)
        if obstacle_x > 0:
            env.robot_position = (obstacle_x - 1, obstacle_y)
            action = 1  # Move down into obstacle
        else:
            env.robot_position = (obstacle_x + 1, obstacle_y)
            action = 0  # Move up into obstacle

        observation, reward, terminated, truncated, info = env.step(action)
        assert reward == -10
        assert terminated is True

# Test the function that ensures target is reachable from robot position
def test_target_is_reachable():
    # Create a layout where the target is surrounded by obstacles
    env = GridEnv(grid_size=(5, 5), num_obstacles=0, fixed_layout=True)
    env.robot_position = (0, 0)
    env.target_position = (2, 2)
    env.obstacle_positions = [(1, 2), (2, 1), (2, 3), (3, 2)]

    # Check if the target is reachable
    assert env._target_is_reachable(env.robot_position, env.target_position, env.obstacle_positions) is False 

    # Now create a layout where the target can be reached
    env.obstacle_positions = [(1, 2), (2, 3), (3, 2)]
    assert env._target_is_reachable(env.robot_position, env.target_position, env.obstacle_positions) is True

    # Also make a 6x6 layout with 5 obstacles to match the required setting
    # One that the target is reachable
    env = GridEnv(grid_size=(6, 6), num_obstacles=5, fixed_layout=False)
    env.robot_position = (0, 0)
    env.target_position = (5, 5)
    env.obstacle_positions = [(1, 0), (1, 1), (2, 1), (3, 3), (4, 4)]
    assert env._target_is_reachable(env.robot_position, env.target_position, env.obstacle_positions) is True

    # One that the target is not reachable
    env.obstacle_positions = [(0, 1), (1, 0), (3,4), (5,1), (4,2)]
    assert env._target_is_reachable(env.robot_position, env.target_position, env.obstacle_positions) is False

if __name__ == "__main__":
    test_reset_returns_correct_info()
    test_step_updates_robot_position()
    test_step_returns_correct_info()
    test_robot_reaches_target()
    test_robot_hits_obstacle()
    test_target_is_reachable()
    print("All tests passed.")