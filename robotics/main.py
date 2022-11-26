from envs import TrainEnv

env = TrainEnv("robot_0")

obs = env.reset()
n_steps = 1_000
for _ in range(n_steps):
    # Random action
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
