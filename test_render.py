from gridbot_env.grid_env import GridEnv
import time

env = GridEnv(render_mode="human")
obs, info = env.reset(seed=42)

done = False
while not done:
    env.render()
    time.sleep(1)
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close()