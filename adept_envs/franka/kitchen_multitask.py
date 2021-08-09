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

ASSET_PATH = 'adept_envs/franka/assets/franka_kitchen.xml'

DEFAULT_OBSERVATION_KEYS = (
    # 't',
    'qp',
    # 'qv',
    'obj_qp',
    'mocap_pos',
    'mocap_quat',
    # 'obj_qv',
    'goal',
)

ELEMENT_INDICES_LL = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Arm
    [9, 10],  # Burners
    [11, 12],  # Burners
    [13, 14],  # Burners
    [15, 16],  # Burners
    [17, 18],  # Lightswitch
    [19],  # Slide
    [20, 21],  # Hinge
    [22],  # Microwave
    [23, 24, 25, 26, 27, 28, 29]  # Kettle
]

ELEMENT_INDICES_HL = [
    [11, 12],  # Bottom Burners
    [15, 16],  # Top Burners
    [17, 18],  # Lightswitch
    [19],  # Slide
    [20, 21],  # Hinge
    [22],  # Microwave
    [23, 24, 25, 26, 27, 28, 29]  # Kettle
]

class KitchenV0(BaseFrankaEnv):
    
    # Number of degrees of freedom of all objects.
    N_DOF_OBJ = 21

    def __init__(self,
                 asset_path: str = ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 40,
                 use_raw_actions: bool = False,
                 camera_settings=dict(
                    distance=2.5,
                    azimuth=66,
                    elevation=-35,),
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
        """
        self.goal = np.zeros((30,))
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            camera_settings=camera_settings,
            sim_backend=SimBackend.DM_CONTROL,
            **kwargs)

        self.use_raw_actions = use_raw_actions
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = self.sim.model.key_qvel[0].copy()
        self.midpoint_pos = np.array([-0.440, 0.152, 2.226])
        self.range = np.array([0.035, 0.035, 0.02])

        self.mocap_pos_clip_lower = np.array([-0.7, 0., 1.8])
        self.mocap_pos_clip_upper = np.array([0.4, 0.5, 2.6])
        # TODO: Configure robot

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        super()._configure_robot(builder)
        # builder.add_group('burners',
        #                   qpos_indices=[11, 12, 15, 16],
        #                   qpos_range=[(-np.pi, np.pi),
        #                               (-np.pi, np.pi),
        #                               (-np.pi, np.pi),
        #                               (-np.pi, np.pi)])

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
            ('qp', np.concatenate([arm_state.qpos, gripper_state.qpos])),
            ('qv', np.concatenate([arm_state.qvel, gripper_state.qvel])),
            ('obj_qp', self.sim.data.qpos[-self.N_DOF_OBJ:]),
            ('obj_qv', self.sim.data.qvel[-self.N_DOF_OBJ:]),
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

class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ms = self.sim.model.site_name2id('microhandle_site')
        self.ee = self.sim.model.site_name2id('end_effector')
        self.microwave_joint_id = -8
        self.microwave_goal_pos = -0.7

    def get_reward_dict(self,
                        action: np.ndarray,
                        obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        ee_error = np.array([-0.64187852, 0.49210206, 1.792]) - self.sim.data.site_xpos[self.ee]
        microwave_error = self.sim.data.qpos[self.microwave_joint_id] - self.microwave_goal_pos
        reward_dict = collections.OrderedDict((
            ('ee_microwave_ee', -1 * np.linalg.norm(ee_error)),
            # ('microwave_pos', -40 * np.linalg.norm(microwave_error)),
        ))
        return reward_dict
