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
    assert new_position != initial_position or initial_position[1] == env.num_cols - 1  # Check if moved or at right edge

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
    assert reward == 1
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
        assert reward == -1
        assert terminated is True
        
if __name__ == "__main__":
    test_reset_returns_correct_info()
    test_step_updates_robot_position()
    test_step_returns_correct_info()
    test_robot_reaches_target()
    test_robot_hits_obstacle()
    print("All tests passed.")