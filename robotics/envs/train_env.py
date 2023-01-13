import os
from gym import utils
from envs import fetch_env


class TrainEnv(fetch_env.FetchEnv, utils.EzPickle):
    """A class for training a member of the genetic algorithm population"""

    def __init__(self, asset_path: str, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        fetch_env.FetchEnv.__init__(
            self,
            os.path.join(asset_path, "complete.xml"),
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.08,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            gripper_joints=[
                "Base/0/0j",
                "Base/0/0j/0j",
                "Base/0/0j/0j/0j",
                "Base/0/1j",
                "Base/0/1j/0j",
                "Base/0/1j/0j/0j",
            ],
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)
