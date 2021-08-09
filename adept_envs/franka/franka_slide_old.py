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
                     elevation=-35, ),
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
        """
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            camera_settings=camera_settings,
            sim_backend=SimBackend.DM_CONTROL,
            **kwargs)
        self.goal = np.zeros(17)

        self.use_raw_actions = use_raw_actions
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = self.sim.model.key_qvel[0].copy()
        self.midpoint_pos = np.array([-0.440, 0.152, 2.226])
        self.range = np.array([0.035, 0.035, 0.02])

        self.mocap_pos_clip_lower = np.array([-0.7, 0., 1.8])
        self.mocap_pos_clip_upper = np.array([0.4, 0.5, 2.7])
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
        for _ in range(10):
            denormalize = True
            new_pos = self.midpoint_pos
            self.sim.data.mocap_pos[:] = new_pos.copy()
            self.robot.step({
                'gripper': np.zeros(2)
            }, denormalize)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        # TODO: How do deal with goal changing?
        denormalize = False if self.use_raw_actions else True
        current_pos = self.sim.data.mocap_pos.copy()
        new_pos = current_pos + action[:3] * self.range
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
            ('qp', np.concatenate([arm_state.qpos, gripper_state.qpos])),
            ('qv', np.concatenate([arm_state.qvel, gripper_state.qvel])),
            ('obj_qp', self.sim.data.qpos[-self.N_DOF_OBJ:]),
            ('mocap_pos', self.sim.data.mocap_pos.copy()),
            ('mocap_quat', self.sim.data.mocap_quat.copy()),
            ('goal', self.goal),
        ))
        return obs_dict

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
            camera = engine.MovableCamera(self.sim, 480, 640)
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
        target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('slide_site')]
        ee_error = target_pos - obs_dict['mocap_pos']
        slider_error = obs_dict['obj_qp'] - 0.4
        reward_dict = collections.OrderedDict((
            ('ee_slider', -1 * np.linalg.norm(ee_error)),
            ('slider', -10 * np.linalg.norm(slider_error)),
        ))
        return reward_dict
#
#
# # 200 goal state
# {'actions': array([0.05756984, -0.1473211, 0.18738647, 1.39521083, -0.05946898,
#                    0.10398416, 0.12969968, 1., 1.]),
#  'obs': array([-2.03817975e+00, -1.50070976e+00, 5.63180499e-01, -1.72759823e+00,
#                2.15470736e+00, 3.81088338e-01, -4.77882720e-01, 3.99906409e-02,
#                3.99951142e-02, -7.13967186e-05, -1.13298409e-01, 3.70760911e-01,
#                2.63018466e+00, 6.53595423e-01, -6.53075155e-01, -2.70360276e-01,
#                -2.70575657e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
#                0.00000000e+00, 0.00000000e+00]),
#  'reward': -4.238930974328816,
#  'done': True,
#  'info': OrderedDict([('obs/t', 58.6400000000257),
#                       ('obs/qp',
#                        array([-2.03817975, -1.50070976, 0.5631805, -1.72759823, 2.15470736,
#                               0.38108834, -0.47788272, 0.03999064, 0.03999511])),
#                       ('obs/qv',
#                        array([-8.49682212e-02, -8.49038270e-04, -1.97728327e-02, 1.70912347e-03,
#                               7.81816387e-02, -4.65657965e-02, -4.41398701e-02, 6.07521963e-05,
#                               -1.58923024e-04])),
#                       ('obs/obj_qp', array([0.278267])),
#                       ('obs/mocap_pos',
#                        array([[-0.11329841, 0.37076091, 2.63018466]])),
#                       ('obs/mocap_quat',
#                        array([[0.65359542, -0.65307515, -0.27036028, -0.27057566]])),
#                       ('reward/ee_slider', -0.2382170071429608),
#                       ('reward/slider', -4.000713967185855),
#                       ('reward/total', -4.238930974328816),
#                       ('TimeLimit.truncated', True)]),
#  'mocap_pos': array([0.32134813, 0.20922703, 0.40721378]),
#  'mocap_rot': array([-0.51471407, 0.45597088, 0.82838952])}
#
# # Goal state
# array([-2.03817975e+00, -1.50070976e+00, 5.63180499e-01, -1.72759823e+00,
#        2.15470736e+00, 3.81088338e-01, -4.77882720e-01, 3.99906409e-02,
#        3.99951142e-02, -7.13967186e-05, -1.13298409e-01, 3.70760911e-01,
#        2.63018466e+00, 6.53595423e-01, -6.53075155e-01, -2.70360276e-01,
#        -2.70575657e-01])
#
# # 400
# array([-2.8720078, -1.61128625, 0.53934544, -1.71605108, 2.91599148,
#        0.02498603, -0.54332063, 0.03995335, 0.03999055, 0.28328401,
#        0.1998587, 0.42310298, 2.59139862, 0.65359542, -0.65307515,
#        -0.27036028, -0.27057566])
#
# {'actions': array([0.22491314, 0.62371978, -0.19030257, 0.98876401, -0.06195882,
#                    0.14422444, 0.03849662, 1., 1.]),
#  'obs': array([-2.8720078, -1.61128625, 0.53934544, -1.71605108, 2.91599148,
#                0.02498603, -0.54332063, 0.03995335, 0.03999055, 0.28328401,
#                0.1998587, 0.42310298, 2.59139862, 0.65359542, -0.65307515,
#                -0.27036028, -0.27057566, 0., 0., 0.,
#                0., 0., 0., 0., 0.,
#                0., 0., 0., 0., 0.,
#                0., 0., 0., 0.]),
#  'reward': -1.3528909166833656,
#  'done': True,
#  'info': OrderedDict([('obs/t', 74.64000000000745),
#                       ('obs/qp',
#                        array([-2.8720078, -1.61128625, 0.53934544, -1.71605108, 2.91599148,
#                               0.02498603, -0.54332063, 0.03995335, 0.03999055])),
#                       ('obs/qv',
#                        array([-0.02328001, -0.03461752, 0.09543875, 0.07788845, 0.04516644,
#                               0.09796584, -0.08872544, 0.00031772, -0.00021532])),
#                       ('obs/obj_qp', array([0.278267])),
#                       ('obs/mocap_pos', array([[0.1998587, 0.42310298, 2.59139862]])),
#                       ('obs/mocap_quat',
#                        array([[0.65359542, -0.65307515, -0.27036028, -0.27057566]])),
#                       ('reward/ee_slider', -0.18573102191036372),
#                       ('reward/slider', -1.1671598947730017),
#                       ('reward/total', -1.3528909166833656),
#                       ('TimeLimit.truncated', True)]),
#  'mocap_pos': array([0.62910179, 0.24711305, 0.37316751]),
#  'mocap_rot': array([-0.5976166, 0.30818185, 1.18944215])}
#
