import gym
import robotics
# env = gym.make('UR5Env-v0')
env = gym.make('PickPlace-v0')
env.reset()

for i in range(300):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # if done:
    #     print("Episode finished after {} timesteps".format(i+1))
    #     break

    env.step(env.action_space.sample())
env.close()
