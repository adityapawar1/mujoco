import numpy as np
import os
from gym import utils, error, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py import MjViewer, functions


class UR5Env(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self):
        utils.EzPickle.__init__(self)

        FILE_PATH = os.path.join(os.path.dirname(__file__),
                                 "./../assets/mujoco-ur5-model/example.xml")
        frame_skip = 5
        mujoco_env.MujocoEnv.__init__(self, FILE_PATH, frame_skip)

    def step(self, a):
        # Carry out one step
        # Don't forget to do self.do_simulation(a, self.frame_skip)
        pass

    def viewer_setup(self):
        # Position the camera
        pass

    def reset_model(self):
        # Reset model to original state.
        # This is called in the overall env.reset method
        # do not call this method directly.
        pass

    def _get_obs(self):
        # Observation of environment fed to agent. This should never be called
        # directly but should be returned through reset_model and step
        pass
