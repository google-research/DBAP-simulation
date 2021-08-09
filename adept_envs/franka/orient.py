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

"""Orientation tasks for Franka arms.

The Franka arm is tasked to match an orientation defined by the environment.
"""

import abc
import collections
from typing import Any, Dict, Optional, Sequence

import numpy as np

from adept_envs.components.robot.timeslicer_robot import TimeSlicerRobotState
from adept_envs.components.builder import ComponentBuilder
from adept_envs.components.robot import RobotState
from adept_envs.franka.base_env import BaseFrankaEnv
from adept_envs.simulation.randomize import SimRandomizer
from adept_envs.utils.configurable import configurable
from adept_envs.utils.resources import get_asset_path

# The observation keys that are concatenated as the environment observation.
DEFAULT_OBSERVATION_KEYS = (
    'qpos',
    'last_action',
    'qpos_error',
)

FRANKA_ASSET_PATH = 'adept_envs/franka/assets/franka_only.xml'


class BaseFrankaOrient(BaseFrankaEnv, metaclass=abc.ABCMeta):
    """Shared logic for Franka Orient tasks."""

    def __init__(self,
                 asset_path: str = FRANKA_ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 40,
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
            **kwargs)

        self._initial_pos = self.sim.model.key_qpos[0].copy()[0:7]
        self._desired_pos = np.array([
            0.43716811, 0.49837533, 0.22739414, -1.3157541, -0.10569798,
            1.86249978, 1.47225368
        ])

    def _configure_robot(self, builder: ComponentBuilder):
        super()._configure_robot(builder)

    def _reset(self):
        """Resets the environment."""
        # Mark the target position in sim.
        self.robot.set_state({
            'arm': RobotState(qpos=self._initial_pos, qvel=np.zeros(7)),
        })

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({'arm': action})

    def get_obs_dict(self) -> Dict[str, Any]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        state = self.robot.get_state('arm')

        obs_dict = collections.OrderedDict((
            ('qpos', state.qpos),
            ('qvel', state.qvel),
            ('last_action', self._get_last_action()),
            ('qpos_error', self._desired_pos[0:7] - state.qpos),
        ))

        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        reward_dict = collections.OrderedDict(
            (('orient_error_cost',
              -1 * np.linalg.norm(obs_dict['qpos_error'])),))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        mean_pos_error = np.mean(np.abs(obs_dict['qpos_error']), axis=1)
        score_dict = collections.OrderedDict((
            # Clip and normalize error to 45 degrees.
            ('points', 1.0 - np.minimum(mean_pos_error / (np.pi / 4), 1)),
            ('success', mean_pos_error < .01),
        ))
        return score_dict

    def _update_overlay(self):
        """Updates the overlay in simulation to show the desired orientation."""
        self.robot.set_state({'overlay': RobotState(qpos=self._desired_pos)})
