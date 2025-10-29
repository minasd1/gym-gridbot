#===================================================================
  # Training using PPO Algorithm for GridBot Environment #
#===================================================================

import argparse
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
                 save_path="models/ppo/", verbose=1, use_checkpoints=False):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_success_rate = 0.0
        self.eval_count = 0
        self.env_kwargs = env_kwargs
        self.use_checkpoints = use_checkpoints # Track mode
        
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
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # Check success based on mode
                if self.use_checkpoints:
                    # Success = reached target with all checkpoints
                    if (terminated and 
                        info["robot_position"] == info["target_position"] and
                        info["current_checkpoint_index"] == len(info["checkpoint_positions"])):
                        successes += 1
                        break
                else:
                    # Success = just reached target
                    if terminated and info["robot_position"] == info["target_position"]:
                        successes += 1
                        break
                
        return successes / self.n_eval_episodes

#===================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train PPO agent for GridBot environment')
    parser.add_argument('--checkpoints', action='store_true', 
                       help='Train with checkpoint environment (Bonus 2)')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Custom save path for models (default: models/ppo/ or models/ppo_checkpoint/)')
    args = parser.parse_args()

    use_checkpoints = args.checkpoints
    
    # Set save path based on mode
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = "models/ppo_checkpoint/" if use_checkpoints else "models/ppo/"
    
    print("\n" + "="*60)
    if use_checkpoints:
        print("Training PPO with CHECKPOINT environment (Bonus 2)")
    else:
        print("Training PPO with BASE environment (Part 3)")
    print("="*60)

    # Define curriculum based on mode
    if use_checkpoints:
        # Checkpoint curriculum: progressive complexity
        curriculum = [
            # Stage 1: Learn basic checkpoint behavior with moderate obstacles
            {"grid_size": (6, 6), "num_obstacles": 3, "num_checkpoints": 1, "timesteps": 500_000},
            
            # Stage 2: Same complexity, but with target obstacle density
            {"grid_size": (6, 6), "num_obstacles": 5, "num_checkpoints": 1, "timesteps": 800_000},

            # Stage 3: Add second checkpoint at target density
            {"grid_size": (6, 6), "num_obstacles": 5, "num_checkpoints": 2, "timesteps": 1_500_000},
        ]
    else:
        # Base environment: single stage with 3M timesteps
        curriculum = [
            {"grid_size": (6, 6), "num_obstacles": 5, "num_checkpoints": 0, "timesteps": 3_000_000},
        ]

    model = None

    # Training loop
    for stage_idx, stage in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"Stage {stage_idx+1}/{len(curriculum)}")
        print(f"Grid: {stage['grid_size']}, Obstacles: {stage['num_obstacles']}", end="")
        if use_checkpoints:
            print(f", Checkpoints: {stage['num_checkpoints']}")
        else:
            print()
        print(f"Training for {stage['timesteps']:,} timesteps")
        print(f"{'='*60}")

        # Set up environment kwargs
        env_kwargs = {
            'grid_size': stage["grid_size"],
            'num_obstacles': stage["num_obstacles"],
            'num_checkpoints': stage["num_checkpoints"],
            'fixed_layout': False,
            'render_mode': None
        }

        # Create training environment
        train_env = make_vec_env(GridEnv, n_envs=8, env_kwargs=env_kwargs, seed=42)
        
        # Create evaluation callback
        eval_cb = RandomLayoutEvalCallback(
            env_kwargs, 
            eval_freq=10000, 
            n_eval_episodes=50, 
            save_path=save_path,
            verbose=1,
            use_checkpoints=use_checkpoints
        )

        if model is None:
            # Create model for first stage
            print("\nCreating new PPO model...")
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=1,
                tensorboard_log="./logs/ppo/",
                device="cpu",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.15,  # Higher exploration
                max_grad_norm=0.5,
            )
        else:
            # Continue with existing model on new environment
            print("\nContinuing training with existing model on new environment...")
            model.set_env(train_env)

        # Train
        print(f"\nStarting training...")
        model.learn(total_timesteps=stage['timesteps'], callback=eval_cb)
        
        print(f"\n Stage {stage_idx+1} complete!")
        print(f"  Best success rate: {eval_cb.best_success_rate:.1%}")

        # Save checkpoint after each stage
        stage_save_path = os.path.join(save_path, f"stage_{stage_idx+1}")
        model.save(stage_save_path)
        print(f"  Model saved to: {stage_save_path}")

        # If final stage and success rate too low (only for checkpoint mode), train more
        if use_checkpoints and stage_idx == len(curriculum) - 1 and eval_cb.best_success_rate < 0.6:
            print("\n Final stage success rate below 60%. Training additional 1M timesteps...")
            model.learn(total_timesteps=1_000_000, callback=eval_cb)
            print(f"  New success rate: {eval_cb.best_success_rate:.1%}")

        train_env.close()
        eval_cb.eval_env.close()

    print("\n" + "=" * 60)
    print(" Training Complete!")
    print("=" * 60)

    # Save final model
    final_save_path = os.path.join(save_path, "ppo_final")
    model.save(final_save_path)
    print(f"\n Final model saved to: {final_save_path}")
    print(f" Best model saved to: {os.path.join(save_path, 'best_model')}")

if __name__ == "__main__":
    main()
