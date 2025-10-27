#===================================================================
  # Training using PPO Algorithm for GridBot Environment #
#===================================================================

from gymnasium.envs.registration import register
register(
    id='GridBot-v0',
    entry_point='gridbot_env.grid_env:GridEnv',
)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gridbot_env.grid_env import GridEnv
import pickle
import os
import numpy as np

# ============================================================
# Custom Evaluation Callback for Random Layouts
# ============================================================
class RandomLayoutEvalCallback(BaseCallback):
    def __init__(self, 
                 eval_freq=10000,    # We evaluate every 10k steps
                 n_eval_episodes=50, # Evaluate over 50 episodes
                 save_path="models/ppo/",
                 verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_success_rate = 0.0    # Keep track of best success rate
        self.eval_count = 0
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Create evaluation environment (random layouts)
        self.eval_env = GridEnv(
            grid_size=(5, 5),
            num_obstacles=1,
            fixed_layout=False,  
            render_mode=None
        )
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            success_rate = self._evaluate_on_random_layouts()
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Evaluation #{self.eval_count} (Step {self.n_calls})")
                print(f"Success rate on random layouts: {success_rate:.1%}")
                print(f"{'='*60}")
            
            # Save if this is the best model so far
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                save_file = os.path.join(self.save_path, "best_model")
                self.model.save(save_file)
                if self.verbose > 0:
                    print(f"New best model saved! Success rate: {success_rate:.1%}\n")
            
        return True
    
    def _evaluate_on_random_layouts(self):
        """Evaluate agent on random layouts"""
        successes = 0
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                
                if terminated and reward > 50:
                    successes += 1
                    break
                    
        return successes / self.n_eval_episodes

# ============================================================
# Main Training Script
# ============================================================
print("=" * 60)
print("Learning for GridBot Using PPO")
print("Training on random layouts from the start")
print("=" * 60)

# Environment configuration
env_kwargs = {
    'grid_size': (5, 5),
    'num_obstacles': 1,
    'fixed_layout': False,  # Train on random layouts
    'render_mode': None
}

print("\nCreating training environment with RANDOM layouts...")
print("Each episode will have a different layout")

# Create training environment
train_env = make_vec_env(
    GridEnv,
    n_envs=8,
    env_kwargs=env_kwargs,
    seed=42
)

# Create evaluation callback
eval_cb = RandomLayoutEvalCallback(
    eval_freq=10000,  # Evaluate every 10k steps
    n_eval_episodes=50,
    save_path="models/ppo/",
    verbose=1
)

# Create PPO model
model = PPO(
    "MlpPolicy", 
    train_env, 
    verbose=1,
    tensorboard_log="./logs/ppo_tensorboard/",
    device="cpu",
    
    # Hyperparameters optimized for generalization
    learning_rate=3e-4,
    n_steps=2048, 
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.1,  
    max_grad_norm=0.5,
)

print("\nStarting training on random layouts...")
print("This approach trains on diverse layouts from the beginning")
print("-" * 60)

# Train
model.learn(
    total_timesteps=1_000_000,  # 1 million steps
    callback=eval_cb
)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Best success rate achieved: {eval_cb.best_success_rate:.1%}")

# Save final model
model.save("models/ppo/ppo_final")
print("\nFinal model saved to: models/ppo/ppo_final.zip")
print("Best model saved to: models/ppo/best_model.zip")

# Cleanup
train_env.close()
eval_cb.eval_env.close()