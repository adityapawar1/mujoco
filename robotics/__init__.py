from gym.envs.registration import register

register(
    id="EndEffectorEnv-v0",
    entry_point="robotics.envs:TrainEnv",
)
