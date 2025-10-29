# gym-gridbot
A Reinforcement Learning environment for 2D grid navigation with obstacles, built using Gymnasium.
Agent learns to navigate a grid, avoid obstacles and reach a set target.

## Project Overview
This project implements:
- **Custom Grid Environment** (`GridEnv`) using Gymnasium
- **PPO** (Proximal Policy Optimization) - on-policy algorithm
- **DQN** (Deep Q-Network) - off-policy algorithm
- **Bonus: Checkpoint Environment** - agents must visit waypoints before reaching the goal
- Training and evaluation scripts for both algorithms

The environment simulates a robot navigating a 6x6 grid with obstacles, learning to reach a target position through reinforcement learning.

## Setup Instructions
```bash
# Make the install script executable
chmod +x install.sh

# Then run the installation
./install.sh
```

After setup, please verify the installation:
```bash
python -c "import gridbot_env; print('Installation successful!')"
```

## Training the Agents

Before training, ensure your virtual environment is activated:
```bash
source venv/bin/activate
```

### Training PPO Agent

#### Base Environment (Part 3 - No Checkpoints)
```bash
python train_ppo.py
```
- Trains on 6x6 grid with 5 obstacles
- 3,000,000 timesteps
- Models saved to `models/ppo/`

#### Checkpoint Environment (Bonus 2)
```bash
python train_ppo.py --checkpoints
```
- Uses curriculum learning (3 stages)
- Progressive difficulty: 3 obstacles/1 checkpoint → 5 obstacles/1 checkpoint → 5 obstacles/2 checkpoints
- Models saved to `models/ppo_checkpoint/`

#### Custom Training Options
```bash
python train_ppo.py --save-path models/custom_ppo/

# Or, for checkpoint environment:
python train_ppo.py --checkpoints --save-path models/my_experiment/
```

### Training DQN Agent (Only for base environment)
```bash
python train_dqn_agent.py
```
- Models saved to `models/dqn/`

## Evaluating the Agents

Before evaluation, ensure your virtual environment is activated:
```bash
source venv/bin/activate
```

The project uses a single evaluation script (`evaluate_agents.py`) to evaluate both PPO and DQN models.

**CAUTION:** Consider also evaluating `ppo_final.zip` and `dqn_final.zip` instead of `best_model.zip`, as it has been experimentally proven that they may perform better. The `best_model.zip` is saved based on intermediate evaluation during training, while `ppo_final.zip`/`dqn_final.zip` represent the final trained models.

### Evaluating PPO - Base Environment (Part 3 - No Checkpoints)
```bash
python evaluate_agents.py
```

**Output Metrics:**
- Average steps per episode 
- Average total reward per episode 
- Success rate (the percentage of episodes where the robot reaches the target)

### Evaluating PPO - Checkpoint Environment (Bonus 2)
```bash
python evaluate_agents.py --checkpoints
```
Default: Loads `models/ppo_checkpoint/best_model.zip` and evaluates over 100 episodes.

**Output Metrics:**
- Success Rate (full): Percentage of episodes where agent reached target AND visited all checkpoints
- Checkpoint Completion Rate: Percentage of episodes where all checkpoints were visited
- Average Checkpoints Visited: Mean number of checkpoints reached per episode
- Checkpoint Distribution: Breakdown of episodes by number of checkpoints visited
- Average Steps per Episode
- Average Total Reward

### Evaluating DQN - Base Environment
```bash
python evaluate_agents.py --model models/dqn/best_model.zip
```

Note: DQN does not support the checkpoint environment.

### Custom Evaluation Options
```bash
# Evaluate a specific PPO model
python evaluate_agents.py --model models/ppo/stage_1.zip

# Run more evaluation episodes
python evaluate_agents.py --episodes 200

# Evaluate PPO checkpoint environment with different number of checkpoints
python evaluate_agents.py --checkpoints --num-checkpoints 3

# Evaluate DQN with more episodes
python evaluate_agents.py --model models/dqn/best_model.zip --episodes 200

# Combined options for PPO checkpoints
python evaluate_agents.py --checkpoints --model models/ppo_checkpoint/stage_2.zip --episodes 150
```

### Command-Line Arguments

- `--checkpoints`: Evaluate on checkpoint environment (only for PPO, default: base environment)
- `--model PATH`: Path to model file (default: `models/ppo/best_model.zip`)
- `--episodes N`: Number of evaluation episodes (default: 100)
- `--num-checkpoints N`: Number of checkpoints in environment (default: 2, only used with `--checkpoints`)