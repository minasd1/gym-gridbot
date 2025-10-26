#===================================================================
  # Train the chosen agents on the GridBot environment #
#===================================================================

# We will use the stable-baselines3 library for training agents
# Chosen agents: PPO (on-policy) and DQN (off-policy)

from gymnasium.envs.registration import register

register(
    id='GridBot-v0',
    entry_point='gridbot_env.grid_env:GridEnv',
)

import gymnasium as gym
from stable_baselines3 import PPO

#===================================================================

model = PPO("MlpPolicy", "GridBot-v0", verbose=1, device="cpu")
model.learn(total_timesteps=100_000)
model.save("models/ppo_gridbot")
