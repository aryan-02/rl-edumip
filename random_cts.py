import gymnasium as gym
import edumip
import numpy as np
import random
import time

import continuous_cartpole
env = gym.make("ContinuousCartPole-v0", render_mode="human")

obs, info = env.reset()

done = False
while not done:
    action = np.random.uniform(low=-1.0, high=1.0, size=(1,)).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    # Sleep for a short duration to visualize the rendering

    print(f"Action: {action}, State: {obs}, Reward: {reward}")
