import gym
from stable_baselines3 import HER
from robotics.envs import TrainEnv
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.env_checker import check_env

import resource
import sys

# resource.setrlimit(
#     resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
# )
sys.setrecursionlimit(10**6)

env = TrainEnv()
# env = gym.make("Walker2d-v2")
check_env(env)

print("Checked env, creating policy")
model = PPO("MultiInputPolicy", env, verbose=2)
print("Starting training")
model.learn(total_timesteps=10_000)
print("Finish training")

obs = env.reset()
n_steps = 1_000
for _ in range(n_steps):
    # Random action
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward)
    if done:
        obs = env.reset()

"""
env = gym.make("Walker2d-v2")

model = HER("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=10_000)

env = gym.make("Walker2d-v2")
obs = env.reset()
done = False
episode_reward = 0

while not done:
    obs, reward, done, _info = env.step(model.predict(obs)[0])
    episode_reward += reward
    env.close()

print("Episode Reward: " + str(episode_reward))

"""
"""
env = TrainEnv()

model = PPO("MultiInputPolicy", env, verbose=2)
model.learn(total_timesteps=10_000)

env = gym.make("EndEffectorEnv-v0")
obs = env.reset()
done = False
episode_reward = 0

while not done:
    obs, reward, done, _info = env.step(model.predict(obs)[0])
    episode_reward += reward
    env.close()

print("Episode Reward: " + str(episode_reward))
"""
