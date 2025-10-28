#===================================================================
  # Training using PPO Algorithm for GridBot Environment #
#===================================================================

from gymnasium.envs.registration import register
register(
    id='GridBot-v0',
    entry_point='gridbot_env.grid_env:GridEnv',
)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from gridbot_env.grid_env import GridEnv
import os

#===================================================================

class RandomLayoutEvalCallback(BaseCallback):
    def __init__(self, env_kwargs, eval_freq=10000, n_eval_episodes=50, 
                 save_path="models/ppo_test/", verbose=1):
        super().__init__(verbose)
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
                    print(f" New best! Success rate: {success_rate:.1%}\n")
            
        return True
    
    def _evaluate_on_random_layouts(self):
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

curriculum = [ # Kept curriculum learning stages for reference
    # {"grid_size": (6, 6), "num_obstacles": 0, "timesteps": 300_000},
    # {"grid_size": (6, 6), "num_obstacles": 1, "timesteps": 400_000},
    # {"grid_size": (6, 6), "num_obstacles": 2, "timesteps": 500_000},
    # {"grid_size": (6, 6), "num_obstacles": 3, "timesteps": 600_000},
    # {"grid_size": (6, 6), "num_obstacles": 4, "timesteps": 800_000},
    {"grid_size": (6, 6), "num_obstacles": 5, "timesteps": 1_500_000},  # Final target
]

model = None

# The structure of the curriculum loop remains for reference
for stage_idx, stage in enumerate(curriculum):
    print(f"\n{'='*60}")
    print(f"Stage {stage_idx+1}/{len(curriculum)}")
    print(f"Grid: {stage['grid_size']}, Obstacles: {stage['num_obstacles']}")
    print(f"Training for {stage['timesteps']:,} timesteps")
    print(f"{'='*60}")

    env_kwargs = {
        'grid_size': stage["grid_size"],
        'num_obstacles': stage["num_obstacles"],
        'fixed_layout': False,
        'render_mode': None
    }

    train_env = make_vec_env(GridEnv, n_envs=8, env_kwargs=env_kwargs, seed=42)
    eval_cb = RandomLayoutEvalCallback(env_kwargs, eval_freq=10000, n_eval_episodes=50, verbose=1)

    if model is None:
        # Create model for first stage
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./logs/ppo_test/",
            device="cpu",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.15,  # Higher exploration for harder task
            max_grad_norm=0.5,
        )
    else:
        # Continue with existing model on new environment
        model.set_env(train_env)

    # Train
    model.learn(total_timesteps=stage['timesteps'], callback=eval_cb)
    
    print(f"\n Stage {stage_idx+1} complete!")
    print(f"  Best success rate: {eval_cb.best_success_rate:.1%}")

    # Save checkpoint
    model.save(f"models/ppo_test/stage_{stage_idx+1}")

    # If final stage and success rate too low, train more
    if stage_idx == len(curriculum) - 1 and eval_cb.best_success_rate < 0.6:
        print("\n Final stage success rate below 60%. Training additional 1M timesteps...")
        model.learn(total_timesteps=1_000_000, callback=eval_cb)
        print(f"  New success rate: {eval_cb.best_success_rate:.1%}")

    train_env.close()
    eval_cb.eval_env.close()

print("\n" + "=" * 60)
print("Curriculum Training Complete!")
print("=" * 60)

model.save("models/ppo_test/ppo_final")
print("\n Final model saved!")
