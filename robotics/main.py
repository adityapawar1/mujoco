import numpy as np
from stable_baselines3.ppo.ppo import PPO
from robotics import utils
from robotics.envs import TrainEnv

chromosome = "0.0456 0.0446 0.0041 2 0.0031 0.0206 0.035100000000000006 2 0.0366 0.0251 0.0176 2 0.041100000000000005 0.0471 0.0476 0 0.0196 0.004600000000000001 0.0391 2 0.0161 0.0071 0.022099999999999998 1 0.0131 0.004600000000000001 0.0026 2 0.0021 0.0371 0.0176 2 0.049600000000000005 0.0306 0.03560000000000001 1 0.0206 0.0206 0.0086 2 0.0031 0.0181 0.0316 1 0.04360000000000001 0.03610000000000001 0.0371 0"
chromosome = map(float, chromosome.split(r" "))

env = TrainEnv()
model = PPO("MultiInputPolicy", env, verbose=2)
model.load("/Users/adityapawar/Downloads/end_effector2")

end_effector = utils.chromosome_to_end_effector(np.array(chromosome), 6)
end_effector.build()

obs = env.reset()
n_steps = 1_000
episode_reward = 0
for _ in range(n_steps):
    # Random action
    env.render()
    obs, reward, done, _ = env.step(model.predict(obs)[0])
    print(reward)
    if done:
        obs = env.reset()
    episode_reward += reward

env.close()
print("Episode Reward: " + str(episode_reward))
