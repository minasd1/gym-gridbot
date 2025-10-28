#===================================================================
  # Evaluate agents using the required metrics #
#===================================================================

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from gridbot_env.grid_env import GridEnv

# ====================================================

# Create environment
env = GridEnv(render_mode=None, grid_size=(6, 6), num_obstacles=5, fixed_layout=False)
model_path = "models/ppo/best_model.zip"

# Load trained model
model = PPO.load(model_path, env=env)

# The requested evaluation metrics
NUM_EPISODES = 100
total_rewards = []
steps_per_episode = []
success_count = 0

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        # Predict action without exploration noise
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    total_rewards.append(total_reward)
    steps_per_episode.append(steps)
    if terminated and info["robot_position"] == info["target_position"]:
        success_count += 1

env.close()

# Compute metrics
avg_steps = np.mean(steps_per_episode)
avg_reward = np.mean(total_rewards)
success_rate = (success_count / NUM_EPISODES) * 100

# Print evaluation results
print("=======================================")
print("Evaluation Results")
print("=======================================")
print(f"Total Episodes:        {NUM_EPISODES}")
print(f"Average Steps:         {avg_steps:.2f}")
print(f"Average Total Reward:  {avg_reward:.2f}")
print(f"Success Rate:          {success_rate:.1f}%")
print("=======================================")
