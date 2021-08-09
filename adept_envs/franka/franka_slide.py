"""
Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Kitchen environment for long horizon manipulation """

import collections
from typing import Dict, Sequence

from dm_control.mujoco import engine
from gym import spaces
import numpy as np

from adept_envs.components.robot import RobotComponentBuilder, RobotState
from adept_envs.franka.base_env import BaseFrankaEnv
from adept_envs.utils.resources import get_asset_path
from adept_envs.simulation.sim_scene import SimBackend

ASSET_PATH = 'adept_envs/franka/assets/franka_slide.xml'

DEFAULT_OBSERVATION_KEYS = (
    'qp',
    'obj_qp',
    'mocap_pos',
    'mocap_quat',
    'goal'
)


class FrankaSlide(BaseFrankaEnv):

    # Number of degrees of freedom of all objects.
    N_DOF_OBJ = 1

    def __init__(self,
                 asset_path: str = ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 40,
                 use_raw_actions: bool = False,
                 camera_settings=dict(
                    distance=2.5,
                    azimuth=66,
                    elevation=-35,),
                 reset_frequency=-1,
                 eval_mode=False,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
        """
        self._eval_mode = eval_mode
        self._reset_frequency = reset_frequency
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            camera_settings=camera_settings,
            sim_backend=SimBackend.DM_CONTROL,
            **kwargs)
        self.goal = np.zeros(10)

        self._goal_idx = -1
        self._goals = [
            # Timestep: slider open
            np.array([0.01998091, 0.02331463, 0.41276523, 0.31941679, 0.41169168,
                      2.53992176, 0.65359542, -0.65307515, -0.27036028, -0.27057566]),
            # Timestep: slider closed
            np.array([4.00591394e-02, 6.13513224e-03, -9.89736586e-04, -1.29648443e-01, 5.00000000e-01,
                      2.57141383e+00, 6.53595423e-01, -6.53075155e-01, -2.70360276e-01, -2.70575657e-01]),
        ]
        self.goal = self._goals[self._goal_idx]
        self.use_raw_actions = use_raw_actions
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = self.sim.model.key_qvel[0].copy()
        self.midpoint_pos = np.array([-0.440, 0.152, 2.226])
        self.range = np.array([0.035, 0.035, 0.02])
        self._reset_count = 0
        self.mocap_pos_clip_lower = np.array([-0.9, 0., 1.8])
        self.mocap_pos_clip_upper = np.array([0.6, 0.5, 2.7])
        # TODO: Configure robot

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        super()._configure_robot(builder)

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        """ If using raw actions, there is no need to do any processing to the action array."""
        if self.use_raw_actions:
            return action
        else:
            return super()._preprocess_action(action)

    def _reset(self):
        pass

    def reset(self):
        """Resets the environment.

        Args:
            state: The state to reset to. This must match with the state space
                of the environment.

        Returns:
            The initial observation of the environment after resetting.
        """
        self.last_action = None
        if self._goal_idx == -1 or self._eval_mode \
                or (self._reset_frequency != -1 and self._reset_count % self._reset_frequency == 0):
            print("Inside the reset")
            self.sim.reset()
            self.sim.forward()
            print("Resetting the environment fully")
            """Resets the environment."""
            self.robot.set_state({
                'arm': RobotState(
                    qpos=self.init_qpos[0:self.N_DOF_ARM],
                    qvel=np.zeros(self.N_DOF_ARM)),
                'gripper': RobotState(
                    qpos=self.init_qpos[self.N_DOF_ARM:self.N_DOF_ARM +
                                        self.N_DOF_GRIPPER],
                    qvel=np.zeros(self.N_DOF_GRIPPER))
            })
            # # Backward
            # self.sim.data.qpos[:] = np.array([-2.91279373, -1.45848572,  0.81272645, -1.67989093,  2.90023844,
            #                                 -0.01661279, -0.84141638,  0.0400003,  0.03998485,  0.42591756])
            # self.sim.data.mocap_pos[:] = np.array([0.33656406, 0.385469, 2.54672835])

            # Forward
            self.sim.data.qpos[:] = np.array([-1.86764219, -1.45153967,  0.77889962, -1.91269807,  2.17284555,
                                                    0.53218614, -0.75415067,  0.0209861 ,  0.02486784,  0.05009629])
            self.sim.data.mocap_pos[:] = np.array([-0.04753155,  0.39993939,  2.55948354])

            for _ in range(100):
                self.sim.step()
            self._goal_idx = -1
        self._goal_idx = (self._goal_idx + 1) % 2
        # Only test how to open the slider
        if self._eval_mode:
            self._goal_idx = 0
        self._reset_count += 1
        print("EVAL MODE IS " + str(self._eval_mode))
        print("Current goal is %d"%self._goal_idx)
        self.goal = self._goals[self._goal_idx]
        obs_dict = self.get_obs_dict()
        self.last_obs_dict = obs_dict
        self.last_reward_dict = None
        self.last_score_dict = None
        self.is_done = False
        self.step_count = 0

        return self._get_obs(obs_dict)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        # TODO: How do deal with goal changing?
        denormalize = False if self.use_raw_actions else True
        current_pos = self.sim.data.mocap_pos.copy()
        new_pos = current_pos + action[:3]*self.range
        new_pos = np.clip(new_pos, self.mocap_pos_clip_lower, self.mocap_pos_clip_upper)
        self.sim.data.mocap_pos[:] = new_pos.copy()
        self.robot.step({
            'gripper': action[-2:]
        }, denormalize)

    def get_obs_dict(self):
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        arm_state = self.robot.get_state('arm')
        gripper_state = self.robot.get_state('gripper')
        # obj_state = self.robot.get_state('object')
        obs_dict = collections.OrderedDict((
            ('t', self.robot.time),
            ('qp', np.concatenate([gripper_state.qpos])),
            ('qv', np.concatenate([gripper_state.qvel])),
            ('obj_qp', self.sim.data.qpos[-self.N_DOF_OBJ:]),
            ('mocap_pos', self.sim.data.mocap_pos.copy()),
            ('mocap_quat', self.sim.data.mocap_quat.copy()),
            ('goal', self.goal),
        ))
        return obs_dict

    def set_goal(self, goal):
        self.goal = goal

    def get_goal(self):
        return self.goal

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        score_dict = collections.OrderedDict()
        return score_dict

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs // 2,))

    def render(self, mode='human'):
        if mode == 'rgb_array':
            camera = engine.MovableCamera(self.sim, 84, 84)
            camera.set_pose(
                distance=2.2, lookat=[-0.2, .5, 2.1], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            super().render()

    def get_reward_dict(self,
                        action: np.ndarray,
                        obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        slider_error = obs_dict['obj_qp'] - self.goal[2] # 3rd element is the slider
        reward_dict = collections.OrderedDict((
            ('ee_slider', np.array([np.float(np.linalg.norm(slider_error) < 0.1)])),
        ))
        return reward_dict