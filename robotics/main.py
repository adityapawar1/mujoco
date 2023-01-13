from envs import TrainEnv
import utils

chromosome = [
    2,
    0.015,
    0.010,
    0.050,
    0.040,
    -0.005,
    -0.005,
    0,
    0.015,
    0.010,
    0.050,
    0.040,
    -0.005,
    -0.005,
    1,
    0.015,
    0.010,
    0.050,
    2,
    0.015,
    0.010,
    0.050,
    0.040,
    0.005,
    0.005,
    0,
    0.015,
    0.010,
    0.050,
    0.040,
    0.005,
    0.005,
    1,
    0.015,
    0.010,
    0.050,
]
end_effector = utils.chromosome_to_end_effector(chromosome, 6)
end_effector.build("main")
env = TrainEnv("main")

obs = env.reset()
n_steps = 1_000
for _ in range(n_steps):
    # Random action
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
