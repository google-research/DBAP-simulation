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

"""Door opening tasks with a Franka arm."""

import collections
from typing import Dict, Sequence

import numpy as np
from adept_envs.components.robot import RobotComponentBuilder, RobotState
from adept_envs.franka.base_env import BaseFrankaEnv
from adept_envs.utils.resources import get_asset_path

DEFAULT_OBSERVATION_KEYS = (
    'franka_qpos',
    'door_qpos',
)

ASSET_PATH = 'adept_envs/franka/assets/franka_kitchen_real.xml'

MOTOR_ID_DOOR = 50

DOOR_RESET_POSE = [0]


class FrankaDoorOpening(BaseFrankaEnv):
    """A simple door opening Task."""

    def __init__(self,
                 asset_path: str = ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 40,
                 **kwargs):
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            **kwargs)

        # self._door_bid = self.model.body_name2id('door')

        self._success_threshold = 0.1

        self._initial_door_pos = 0
        self._initial_door_vel = 0
        self._target_door_pos = np.pi / 2

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        super()._configure_robot(builder)
        # Add the door group.
        builder.add_group(
            'door',
            qpos_indices=[-1],  # The object is the last qpos.
            qpos_range=[(-np.pi / 2, 0)])
        if self._device_path:
            builder.update_group('door', motor_ids=[MOTOR_ID_DOOR])

    def _reset(self):
        """Resets the environment."""
        #self._reset_door(DOOR_RESET_POSE)

        franka_init_state, door_init_state = self.robot.get_initial_state(
            ['arm', 'door'])
        franka_pos = franka_init_state.qpos
        door_pos = door_init_state.qpos

        # Eventually there will probably need to be a custom reset procedure for the physical robot.
        self.robot.set_state({
            'arm': RobotState(qpos=franka_pos, qvel=0),
            'door': RobotState(qpos=door_pos, qvel=0),
        })

        # Disengage the motor.
        # if self._interactive and self.robot.is_hardware:
        #         self.robot.set_motors_engaged('door', False)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({'arm': action})

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        franka_state, door_state = self.robot.get_state(['arm', 'door'])

        # Calculate the signed angle difference to the target door orientation [0, +pi/2]
        target_error = self._target_door_pos - door_state.qpos

        obs_dict = collections.OrderedDict(
            (('franka_qpos', franka_state.qpos), ('door_qpos', door_state.qpos),
             ('target_error', target_error)))

        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        reward_dict = collections.OrderedDict(
            (('target_dist_cost', -1 * np.abs(obs_dict['target_error'])),))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        target_dist = np.abs(obs_dict['target_error'])
        score_dict = collections.OrderedDict((
            ('points', 1.0 - target_dist / np.pi),
            ('success', target_dist < self._success_threshold),
        ))
        return score_dict
