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

"""Shared logic for all Franka environments."""

import abc
from typing import Dict, Optional, Sequence, Union

import gym
import numpy as np
from adept_envs.components.robot import RobotComponentBuilder, RobotState
from adept_envs.robot_env import make_box_space, RobotEnv

TINY = 1e-8

class BaseFrankaEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all Franka robot tasks."""

    N_DOF_ARM = 7
    N_DOF_GRIPPER = 2

    def __init__(self,
                 *args,
                 device_path: Optional[str] = None,
                 is_hardware: Optional[bool] = False,
                 **kwargs):
        """Initializes the environment."""
        super().__init__(*args, **kwargs)

        self._device_path = device_path
        self._is_hardware = is_hardware

        # Configure the robot component.
        robot_builder = RobotComponentBuilder()
        self._configure_robot(robot_builder)

        # Create the components.
        self.robot = self._add_component(robot_builder)

    def normalize_control(self, control):
        """ Maps control values to [-1, 1] based on defined limits """
        config = self.robot.get_config('arm')
        norm_control = control / (config.denormalize_range + TINY)
        norm_control = np.clip(norm_control, -1.0, 1.0)
        return norm_control

    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns the current state of the environment."""
        state = self.robot.get_state('arm')
        return {'qpos': state.qpos, 'qvel': state.qvel}

    def set_state(self, state: Dict[str, np.ndarray]):
        """Sets the state of the environment."""
        self.robot.set_state(
            {'arm': RobotState(qpos=state['qpos'], qvel=state['qvel'])})

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        builder.add_group(
            'arm',
            qpos_indices=range(7),
            qpos_range=[
                # Values used previously, by Abishek in master branch
                # (-2.9, 2.9),
                # (-1.8, 1.8),
                # (-2.9, 2.9),
                # (-3.1, 0),
                # (-2.9, 2.9),
                # (00.0, 3.8),
                # (-2.9, 2.9),
                # These are the actual values
                (-2.8973, 2.8973),
                (-1.7628, 1.7628),
                (-2.8973, 2.8973),
                (-3.0718, 0.0698),
                (-2.8973, 2.8973),
                (-0.0175, 3.7525),
                (-2.8973, 2.8973),
            ],
            qvel_range=[
                # Values used previously, by Abishek in master branch
                # (-10, 10),
                # (-10, 10),
                # (-10, 10),
                # (-10, 10),
                # (-10, 10),
                # (-10, 10),
                # (-10, 10),
                # These are the actual values
                (-2.1750, 2.1750),
                (-2.1750, 2.1750),
                (-2.1750, 2.1750),
                (-2.1750, 2.1750),
                (-2.6100, 2.6100),
                (-2.6100, 2.6100),
                (-2.6100, 2.6100)
            ])
        builder.add_group(
            'gripper',
            qpos_indices=[7, 8],
            qpos_range=[
                (0.00, 0.04),
                (0.00, 0.04),
            ],
            actuator_indices=[0, 1]
        )
        if self._is_hardware:
            builder.update_group('arm', part_name='arm')
            builder.update_group('gripper', part_name='gripper')
            builder.set_timeslicer_robot()

    def _initialize_action_space(self) -> gym.Space:
        """Returns the observation space to use for this environment."""
        arm_qpos_indices = self.robot.get_config('arm').qpos_indices
        gripper_qpos_indices = self.robot.get_config('gripper').qpos_indices
        return make_box_space(
            -1.0,
            1.0,
            shape=(arm_qpos_indices.size + gripper_qpos_indices.size,))
