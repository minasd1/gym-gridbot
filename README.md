# gym-gridbot
A Reinfocement Learning environment for 2D grid navigation with obstacles, built using Gymnasium.
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
# Make the install script executable
chmod +x install.sh
# Then run the installation
./install.sh

After setup, please verify the installation:
python -c "import gridbot_env; print('Installation successful!')"

## Training the agents

Before training, ensure your virtual environment is activated:
source venv/bin/activate

## Training PPO agent

#### Base Environment (Part 3 - No Checkpoints)
- Trains on 6x6 grid with 5 obstacles
- 3,000,000 timesteps
- Models saved to `models/ppo/`

#### Checkpoint Environment (Bonus 2)
- Uses curriculum learning (3 stages)
- Progressive difficulty: 3 obstacles/1 checkpoint → 5 obstacles/1 checkpoint → 5 obstacles/2 checkpoints
- Models saved to `models/ppo_checkpoint/`

  #### Custom Training Options
  python train_ppo.py --save-path models/custom_ppo/
  Or, for checkpoint environment:
  python train_ppo.py --checkpoints --save-path models/my_experiment/

  ### Training DQN Agent (Only for base environment)
  python train_dqn_agent.py


  
