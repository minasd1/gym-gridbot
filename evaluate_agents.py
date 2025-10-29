#===================================================================
  # Evaluate PPO Agent on GridBot Environment #
#===================================================================
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from gridbot_env.grid_env import GridEnv

#===================================================================
# An evaluation function for the base environment - the initial requirement
def evaluate_base_environment(model_path, num_episodes=100):
    """Evaluate agent on base environment (no checkpoints)"""
    
    print("="*60)
    print("Base Environment Evaluation")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Environment: 6x6 grid, 5 obstacles")
    print("="*60)
    
    # Create environment
    env = GridEnv(render_mode=None, grid_size=(6, 6), num_obstacles=5, 
                  num_checkpoints=0, fixed_layout=False)
    
    # Load model
    print(f"\nLoading model...")

    if "dqn" in model_path.lower():
        model = DQN.load(model_path, env=env)
        print("DQN model loaded successfully!")
    else:
        model = PPO.load(model_path, env=env)
        print("PPO model loaded successfully!")
    
    # Evaluation metrics
    total_rewards = []
    steps_per_episode = []
    success_count = 0
    
    print(f"\nEvaluating over {num_episodes} episodes...")
    print("-"*60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        total_rewards.append(total_reward)
        steps_per_episode.append(steps)
        
        # Check success
        if terminated and info["robot_position"] == info["target_position"]:
            success_count += 1
        
        # Progress indicator
        if (episode + 1) % 20 == 0:
            current_success = (success_count / (episode + 1)) * 100
            print(f"Progress: {episode + 1}/{num_episodes} episodes | "
                  f"Success rate so far: {current_success:.1f}%")
    
    env.close()
    
    # Compute metrics
    avg_steps = np.mean(steps_per_episode)
    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / num_episodes) * 100
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total Episodes:        {num_episodes}")
    print(f"Success Rate:          {success_rate:.1f}%")
    print(f"Average Steps:         {avg_steps:.2f}")
    print(f"Average Total Reward:  {avg_reward:.2f}")
    print("="*60)
    
    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward
    }

# An evaluation function for the checkpoint environment
def evaluate_checkpoint_environment(model_path, num_episodes=100, num_checkpoints=2):
    """Evaluate agent on checkpoint environment"""
    
    print("="*60)
    print("Checkpoint Environment Evaluation")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Environment: 6x6 grid, 5 obstacles, {num_checkpoints} checkpoints")
    print("="*60)
    
    # Create environment
    env = GridEnv(render_mode=None, grid_size=(6, 6), num_obstacles=5, 
                  num_checkpoints=num_checkpoints, fixed_layout=False)
    
    # Load model
    print(f"\nLoading model...")
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully!")
    
    # Evaluation metrics
    total_rewards = []
    steps_per_episode = []
    success_count = 0
    checkpoints_per_episode = []
    
    print(f"\nEvaluating over {num_episodes} episodes...")
    print("-"*60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Collect stats
        checkpoints_reached = len(info["checkpoints_visited"])
        total_checkpoints = env.num_checkpoints
        checkpoints_per_episode.append((checkpoints_reached, total_checkpoints))
        total_rewards.append(total_reward)
        steps_per_episode.append(steps)
        
        # Check if truly successful
        if (terminated and 
            info["robot_position"] == info["target_position"] and
            info["current_checkpoint_index"] == total_checkpoints):
            success_count += 1
        
        # Progress indicator
        if (episode + 1) % 20 == 0:
            current_success = (success_count / (episode + 1)) * 100
            print(f"Progress: {episode + 1}/{num_episodes} episodes | "
                  f"Success rate so far: {current_success:.1f}%")
    
    env.close()
    
    # Compute metrics
    avg_steps = np.mean(steps_per_episode)
    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / num_episodes) * 100
    
    # Checkpoint analysis
    avg_checkpoints = np.mean([cp[0] for cp in checkpoints_per_episode])
    all_checkpoints_visited = sum(1 for cp, total in checkpoints_per_episode 
                                   if cp == total)
    checkpoint_completion_rate = (all_checkpoints_visited / num_episodes) * 100
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results - Checkpoint Environment")
    print("="*60)
    print(f"Total Episodes:                {num_episodes}")
    print(f"Success Rate (full):           {success_rate:.1f}%")
    print(f"  (reached target + all CPs)")
    print(f"Checkpoint Completion Rate:    {checkpoint_completion_rate:.1f}%")
    print(f"  (visited all checkpoints)")
    print(f"Average Checkpoints Visited:   {avg_checkpoints:.2f} / {num_checkpoints}")
    print("-"*60)
    print(f"Average Steps per Episode:     {avg_steps:.2f}")
    print(f"Average Total Reward:          {avg_reward:.2f}")
    print("="*60)
    
    # Detailed checkpoint breakdown
    checkpoint_distribution = {}
    for cp_count, total in checkpoints_per_episode:
        if cp_count not in checkpoint_distribution:
            checkpoint_distribution[cp_count] = 0
        checkpoint_distribution[cp_count] += 1
    
    print("\nCheckpoint Distribution:")
    print("-"*60)
    for cp_count in sorted(checkpoint_distribution.keys()):
        count = checkpoint_distribution[cp_count]
        percentage = (count / num_episodes) * 100
        print(f"  {cp_count} checkpoints visited: {count} episodes ({percentage:.1f}%)")
    print("="*60)
    
    return {
        'success_rate': success_rate,
        'checkpoint_completion_rate': checkpoint_completion_rate,
        'avg_checkpoints': avg_checkpoints,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward
    }

#===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PPO agent on GridBot environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate base environment
  python evaluate_ppo.py
  
  # Evaluate checkpoint environment
  python evaluate_ppo.py --checkpoints
  
  # Custom model path
  python evaluate_ppo.py --model models/ppo/stage_1.zip
  
  # Custom number of episodes
  python evaluate_ppo.py --checkpoints --episodes 200
        """
    )
    
    parser.add_argument('--checkpoints', action='store_true',
                       help='Evaluate on checkpoint environment (Bonus 2)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: models/ppo/ppo_final.zip or models/ppo_checkpoint/ppo_final.zip)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--num-checkpoints', type=int, default=2,
                       help='Number of checkpoints in environment (default: 2, only used with --checkpoints)')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        if args.checkpoints:
            model_path = "models/ppo_checkpoint/ppo_final.zip"
        else:
            model_path = "models/ppo/ppo_final.zip"
    
    # Run evaluation
    if args.checkpoints:
        results = evaluate_checkpoint_environment(
            model_path, 
            num_episodes=args.episodes,
            num_checkpoints=args.num_checkpoints
        )
    else:
        results = evaluate_base_environment(
            model_path,
            num_episodes=args.episodes
        )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()