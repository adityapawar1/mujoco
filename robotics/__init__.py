from gym.envs.registration import register
register(id='UR5Env-v0',
         entry_point='robotics.envs:UR5Env',)
register(id='PickPlace-v0',
         entry_point='robotics.envs:FetchPickAndPlaceEnv',)
