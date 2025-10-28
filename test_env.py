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
    assert "checkpoint_positions" in info
    assert len(info["obstacle_positions"]) == env.num_obstacles
    assert len(info["checkpoint_positions"]) == env.num_checkpoints

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
    assert "checkpoint_positions" in info

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

# Test robot reaching a checkpoint - the reward should be 50 and the episode should not terminate
def test_robot_reaches_checkpoint():
    env = GridEnv()
    env.reset(seed=42)

    if env.checkpoint_positions:
        checkpoint_x, checkpoint_y = env.checkpoint_positions[0]

        # Place robot near checkpoint (just above it)
        if checkpoint_x > 0:
            env.robot_position = (checkpoint_x - 1, checkpoint_y)
            action = 1  # Move down into checkpoint
        else:
            env.robot_position = (checkpoint_x + 1, checkpoint_y)
            action = 0  # Move up into checkpoint

        observation, reward, terminated, truncated, info = env.step(action)
        print("Checkpoint test: reward =", reward, "terminated =", terminated)
        assert reward == 50
        assert terminated is False

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

# Test that none of the entities overlap in a generated layout
def test_no_entity_overlap_in_layout():
    env = GridEnv(grid_size=(6, 6), num_obstacles=5, num_checkpoints=2,fixed_layout=False)
    robot_pos, obstacle_positions, checkpoint_positions, target_pos = env._generate_layout()

    # Check that robot does not overlap with obstacles or target
    assert robot_pos != target_pos
    assert robot_pos not in obstacle_positions
    assert robot_pos not in checkpoint_positions

    # Check that target does not overlap with obstacles
    assert target_pos not in checkpoint_positions
    assert target_pos not in obstacle_positions

# Test that each entity is correctly represented in the observation channels
def test_observation_channels():
    env = GridEnv(grid_size=(6, 6), num_obstacles=5, num_checkpoints=2, fixed_layout=False)
    observation, info = env.reset(seed=42)

    robot_pos = env.robot_position
    target_pos = env.target_position
    obstacle_positions = env.obstacle_positions
    checkpoint_positions = env.checkpoint_positions

    # Check robot channel (Channel 0)
    assert observation[robot_pos[0], robot_pos[1], 0] == 1.0
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            if (i, j) != robot_pos:
                assert observation[i, j, 0] == 0.0

    # Check obstacle channel (Channel 1)
    for obs_pos in obstacle_positions:
        assert observation[obs_pos[0], obs_pos[1], 1] == 1.0
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            if (i, j) not in obstacle_positions:
                assert observation[i, j, 1] == 0.0

    # Check checkpoint channel (Channel 2)
    for chk_pos in checkpoint_positions:
        assert observation[chk_pos[0], chk_pos[1], 2] == 1.0
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            if (i, j) not in checkpoint_positions:
                assert observation[i, j, 2] == 0.0

    # Check target channel (Channel 3)
    assert observation[target_pos[0], target_pos[1], 3] == 1.0
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            if (i, j) != target_pos:
                assert observation[i, j, 3] == 0.0

    # Check free space channel (Channel 4)
    for i in range(env.grid_size[0]):
        for j in range(env.grid_size[1]):
            if (i, j) == robot_pos or (i, j) in obstacle_positions or (i, j) in checkpoint_positions or (i, j) == target_pos:
                assert observation[i, j, 4] == 0.0 

if __name__ == "__main__":
    test_reset_returns_correct_info()
    test_step_updates_robot_position()
    test_step_returns_correct_info()
    test_robot_reaches_target()
    test_robot_hits_obstacle()
    test_robot_reaches_checkpoint()
    test_target_is_reachable()
    test_no_entity_overlap_in_layout()
    test_observation_channels()
    print("All tests passed.")