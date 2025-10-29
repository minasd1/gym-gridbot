#===================================================================
  # Training using DQN Algorithm for GridBot Environment #
#===================================================================

from gymnasium.envs.registration import register
register(
    id='GridBot-v0',
    entry_point='gridbot_env.grid_env:GridEnv',
)

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gridbot_env.grid_env import GridEnv
import os

#===================================================================

# Same as PPO callback
class RandomLayoutEvalCallback(BaseCallback):
    def __init__(self, env_kwargs, eval_freq=10000, n_eval_episodes=50, 
                 save_path="models/dqn/", verbose=0):
        super().__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_success_rate = 0.0
        self.eval_count = 0
        self.env_kwargs = env_kwargs
        
        os.makedirs(save_path, exist_ok=True)
        self.eval_env = GridEnv(**env_kwargs)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            success_rate = self._evaluate_on_random_layouts()
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Evaluation #{self.eval_count} (Step {self.n_calls})")
                print(f"Success rate: {success_rate:.1%}")
                print(f"{'='*60}")
            
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                save_file = os.path.join(self.save_path, "best_model")
                self.model.save(save_file)
                if self.verbose > 0:
                    print(f"✨ New best! Success rate: {success_rate:.1%}\n")
            
        return True
    
    def _evaluate_on_random_layouts(self):
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                if terminated and info["robot_position"] == info["target_position"]:
                    successes += 1
                    break
        return successes / self.n_eval_episodes

env_kwargs = {
    'grid_size': (6, 6),
    'num_obstacles': 5,
    'fixed_layout': False,
    'render_mode': None
}

# Create vectorized environment
train_env = make_vec_env(GridEnv, n_envs=8, env_kwargs=env_kwargs, seed=42)

# Create evaluation callback
eval_cb = RandomLayoutEvalCallback(
    env_kwargs, 
    eval_freq=10000, 
    n_eval_episodes=50, 
    save_path="models/dqn/",
    verbose=0
)

print("\n Creating DQN model...")

model = DQN(
    "MlpPolicy",
    train_env,
    verbose=0,
    tensorboard_log="./logs/dqn/",
    device="cpu", 
    
    learning_rate=3e-4,           # 3 times higher (was 1e-4)
    buffer_size=200_000,          # 2 times larger (was 100k)
    learning_starts=20_000,       # More initial exploration (was 10k)
    batch_size=64,                # Smaller = more frequent updates (was 128)
    
    tau=0.005,                    # Soft updates (was 1.0)
    gamma=0.99,
    train_freq=4,
    gradient_steps=2,             # 2 times learning per update (was 1)
    target_update_interval=500,   # More frequent (was 1000)
    
    # More exploration throughout training:
    exploration_fraction=0.5,     # Explore longer (was 0.3)
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,    # 2 times more (was 0.05)

    policy_kwargs=dict(
        net_arch=[64, 64]  # Two hidden layers with 64 neurons each
    ),
)

print(" DQN model created!")
print(f"\n Training Configuration:")
print(f"   Environment: 6×6 grid with 5 obstacles")
print(f"   Parallel envs: 8")
print(f"   Total timesteps: 3,000,000")
print(f"   Replay buffer: 200,000")
print(f"   Exploration: 50% of training (1,500,000 steps)")
print(f"   Evaluation: Every 10,000 steps (50 episodes)")


# Train 
print(f"\n{'='*60}")
print(" Starting Training...")
print(f"{'='*60}\n")

model.learn(
    total_timesteps=3_000_000,  # Double the timesteps (was 1.5M)
    callback=eval_cb,
    log_interval=10
)

print(f"\n{'='*60}")
print(" Training Complete!")
print(f"{'='*60}")
print(f" Best success rate achieved: {eval_cb.best_success_rate:.1%}")
print(f" Best model saved to: models/dqn/best_model.zip")

model.save("models/dqn/dqn_final")
print(" Final model saved to: models/dqn/dqn_final.zip")

train_env.close()
eval_cb.eval_env.close()