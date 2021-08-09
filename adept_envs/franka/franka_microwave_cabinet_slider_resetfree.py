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
import pickle
ASSET_PATH = 'adept_envs/franka/assets/franka_microwave_cabinet_slider.xml'
import gym
DEFAULT_OBSERVATION_KEYS = (
    'qp',
    'obj_qp',
    'mocap_pos',
    # 'mocap_quat',
    'goal'
)
import sys
sys.path.append(".")
from rlkit.torch.networks import ConcatMlp, Mlp
import torch
import torch.nn as nn

class FrankaMicrowaveCabinetSlider(BaseFrankaEnv):

    # Number of degrees of freedom of all objects.
    N_DOF_OBJ = 4

    def __init__(self,
                 asset_path: str = ASSET_PATH,
                 observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
                 frame_skip: int = 40,
                 use_raw_actions: bool = False,
                 camera_settings=dict(
                    distance=2.5,
                    azimuth=66,
                    elevation=-35,),
                 eval_mode=False,
                 attempt_limit=50,
                 reset_frequency=-1,
                 idx_completion=False,
                 learned_model=False,
                 learned_model_path=None,
                 counts_enabled=False,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
        """
        self._eval_mode = eval_mode
        self.reset_counter = 0
        self._reset_frequency = reset_frequency
        self._idx_completion = idx_completion
        self.current_idx = 0
        self._counts_enabled = counts_enabled
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            camera_settings=camera_settings,
            sim_backend=SimBackend.DM_CONTROL,
            **kwargs)
        self.commanded_start = -1
        self.commanded_goal = -1
        self.goal = np.zeros(10)
        self.use_raw_actions = use_raw_actions
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = self.sim.model.key_qvel[0].copy()
        self.labeled_goals = pickle.load(open('sim_slider_cabinet_labeled_goals.pkl', 'rb'))
        self.adjacency_matrix = pickle.load(open('sim_slider_cabinet_adjacency_matrix.pkl', 'rb'))
        self._counts = np.zeros(self.adjacency_matrix.shape[0])
        self.midpoint_pos = np.array([-0.440, 0.152, 2.226])
        self.range = np.array([0.035, 0.035, 0.02])
        self.attempt_counter = 0
        self.attempt_limit = attempt_limit
        self.mocap_pos_clip_lower = np.array([-0.85, 0., 1.8])
        self.mocap_pos_clip_upper = np.array([0.55, 0.5, 2.7])
        # TODO: Configure robot
        self.learned_model = learned_model
        self.learned_model_path = learned_model_path
        self.model = None
        if self.learned_model:
            self.model = Mlp(input_size=4,
                            output_size=4,
                            hidden_sizes=(256, 256, 256))
            dat = torch.load(self.learned_model_path)
            state_dict = dat.state_dict()
            self.model.load_state_dict(state_dict)

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(5,))

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
        if self.attempt_counter >= self.attempt_limit:
            self.reset_counter = 0

        self.last_action = None
        # self.sim.reset()
        # self.sim.forward()
        # """Resets the environment."""
        # self.robot.set_state({
        #     'arm': RobotState(
        #         qpos=self.init_qpos[0:self.N_DOF_ARM],
        #         qvel=np.zeros(self.N_DOF_ARM)),
        #     'gripper': RobotState(
        #         qpos=self.init_qpos[self.N_DOF_ARM:self.N_DOF_ARM +
        #                             self.N_DOF_GRIPPER],
        #         qvel=np.zeros(self.N_DOF_GRIPPER))
        # })
        # Choose a random state from labeled goals as the reset state
        if self._eval_mode or self.reset_counter == 0 or \
                (self._reset_frequency != -1 and self.reset_counter % self._reset_frequency == 0):
            print("Resetting the environment fully")
            if self.commanded_start == -1:
                curr_goal_idx = np.random.randint(4)
            else:
                curr_goal_idx = self.commanded_start
            print("RESET TO GOAL POSITION", curr_goal_idx)
            li = 0 #np.random.choice(np.arange(len(self.labeled_goals[curr_goal_idx])))
            curr_goal = self.labeled_goals[curr_goal_idx][li]
            # Choose a random state from next state in relabeled goals as the goal
            # Forward
            new_qpos = np.zeros(13)
            new_qpos[:7] = np.array([-2.64311209, -1.76372997, -0.23182923, -2.1470029 ,  2.55216266, -0.44102682, -0.01343831])
            new_qpos[7:9] = np.array([0.1, 0.1])
            new_qpos[9:] = curr_goal[2:6]
            self.sim.data.qpos[:] = new_qpos.copy()
            self.sim.data.mocap_pos[:] = curr_goal[6:9]
            for _ in range(100):
                self.sim.step()
                self.robot.step({
                    'gripper': 1*np.ones(2)
                }, True)
            if self.commanded_goal == -1:
                next_index = np.random.choice(np.where(self.adjacency_matrix[curr_goal_idx] > 0)[0])
            else:
                next_index = self.commanded_goal
            next_li = 0 #np.random.choice(np.arange(len(self.labeled_goals[next_index])))
            self.goal = np.ones((10,))*next_index #self.labeled_goals[next_index][next_li][:13]
            self.goal_idx = next_index
            self.attempt_counter = 0
            print("NEXT_GOAL", next_index)
        else:
            print("Not resetting")
            # TODO: Check if current goal is accomplished
            if self.check_goal_completion(self.get_obs_dict()['obj_qp']) == self.goal_idx:
                curr_goal_idx = self.goal_idx
                next_index = np.random.choice(np.where(self.adjacency_matrix[curr_goal_idx] > 0)[0])
                next_index = self.learned_goal_select(next_index)
                next_li = np.random.choice(np.arange(len(self.labeled_goals[next_index])))
                self.goal = np.ones((10,))*next_index #self.labeled_goals[next_index][next_li][:13]
                self.goal_idx = next_index
                self.attempt_counter = 0
                print("GOING TO GOAL %d"%self.goal_idx)
            else:
                self.attempt_counter += 1   

            # Move arm back to the middle
            obj_qp = self.get_obs_dict()['obj_qp'].copy()
            curr_goal_idx = self.goal_idx
            li = np.random.choice(np.arange(len(self.labeled_goals[curr_goal_idx])))
            curr_goal = self.labeled_goals[curr_goal_idx][li]
            # Choose a random state from next state in relabeled goals as the goal
            # Forward
            new_qpos = np.zeros(13)
            new_qpos[:7] = np.array([-2.64311209, -1.76372997, -0.23182923, -2.1470029 ,  2.55216266, -0.44102682, -0.01343831])
            new_qpos[7:9] = np.array([0.1, 0.1])
            new_qpos[9:] = obj_qp
            self.sim.data.qpos[:] = new_qpos.copy()
            self.sim.data.mocap_pos[:] = curr_goal[6:9]
            for _ in range(100):
                self.sim.step()
                self.robot.step({
                    'gripper': 1*np.ones(2)
                }, True)
            # else keep going with current goal

        obs_dict = self.get_obs_dict()
        self.last_obs_dict = obs_dict
        self.last_reward_dict = None
        self.last_score_dict = None
        self.is_done = False
        self.step_count = 0
        self.reset_counter += 1
        self.current_idx = self.check_goal_completion(self.get_obs_dict()['obj_qp'][None,:].squeeze(axis=0))
        return self._get_obs(obs_dict)

    def learned_goal_select(self, goal_selected):
        if self.learned_model:
            print("IN LEARNED MODEL")
            o = self._get_obs(self.get_obs_dict())[2:6]
            input_x = torch.Tensor(o)[None, :]
            output_x = torch.nn.Softmax()(self.model(input_x)*np.exp(-self._counts)).detach().numpy()[0]
            goal_selected = np.random.choice(range(4), p=output_x)
            print("LEARNED LIKELIHOOD PREDICTIONS " + str(output_x))

        # Updating counts
        curr_count = np.zeros((self._counts.shape[0],))
        curr_count[goal_selected] += 1
        self.update_counts(curr_count)

        return goal_selected

    def update_counts(self, new_counts):
        if self._counts_enabled:
            self._counts += new_counts

    def check_goal_completion(self, curr_pos):
        max_objs = np.array([0.17, 1, 0.6, -0.05])
        min_objs = np.array([0.08, 0.1, 0.2, -0.2])
        init_bitflips = np.array([0, 0, 0, 1])
        curr_bitflips = init_bitflips.copy()
        for j in range(4):
            if curr_pos[j] > max_objs[j]:
                curr_bitflips[j] = 1
            elif curr_pos[j] < min_objs[j]:
                curr_bitflips[j] = 0
        new_idx = 2 * curr_bitflips[0] + curr_bitflips[2]
        return new_idx

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        # TODO: How do deal with goal changing?
        denormalize = False if self.use_raw_actions else True
        current_pos = self.sim.data.mocap_pos.copy()
        meanval = (self.mocap_pos_clip_upper + self.mocap_pos_clip_lower)/2.0
        rng = (self.mocap_pos_clip_upper - self.mocap_pos_clip_lower)/2.0
        # new_pos = action[:3]*rng + meanval #current_pos + action[:3]*self.range
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
        # TODO: Check if the goal index is satisfied.
        max_delta_slider = 0.22874171 
        max_delta_cabinet = 1.01982685
        if not self._idx_completion:
            g = self.labeled_goals[self.goal_idx][0]
            if self.goal_idx == 0:
                if self.current_idx == 2 or self.current_idx == 0:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('slide_site')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 0:1] - g[2:3])/max_delta_slider
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                elif self.current_idx == 1 or self.current_idx == 3:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('hinge_site2')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 2:3] - g[4:5])/max_delta_cabinet
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                return reward_dict
            elif self.goal_idx == 1:
                if self.current_idx == 1 or self.current_idx == 3:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('slide_site')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 0:1] - g[2:3])/max_delta_slider
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                elif self.current_idx == 2 or self.current_idx == 0:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('hinge_site2')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 2:3] - g[4:5])/max_delta_cabinet
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                return reward_dict
            elif self.goal_idx == 2:
                if self.current_idx == 0 or self.current_idx == 2:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('slide_site')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 0:1] - g[2:3])/max_delta_slider
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                elif self.current_idx == 1 or self.current_idx == 3:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('hinge_site2')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 2:3] - g[4:5])/max_delta_cabinet
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                return reward_dict
            elif self.goal_idx == 3:
                if self.current_idx == 1 or self.current_idx == 3:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('slide_site')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 0:1] - g[2:3])/max_delta_slider
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                elif self.current_idx == 0 or self.current_idx == 2:
                    target_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('hinge_site2')]
                    arm_error = obs_dict['mocap_pos'] - target_pos
                    slider_error = (obs_dict['obj_qp'][:, 2:3] - g[4:5])/max_delta_cabinet
                    reward_dict = collections.OrderedDict((
                        ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))])),
                        ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
                    ))
                return reward_dict
            else:
                raise Exception("Wrong index")
        else:
            current_idx = self.check_goal_completion(obs_dict['obj_qp'].squeeze(axis=0))
            reward_dict = collections.OrderedDict((
                ('completion', np.array([np.float(current_idx == self.goal_idx)])),
            ))
            return reward_dict