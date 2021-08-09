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
import pathlib
path_orig = pathlib.Path(__file__).parent.absolute()
import collections
from typing import Dict, Sequence
import gym
from dm_control.mujoco import engine
from gym import spaces
import numpy as np
import copy
import torch
from rlkit.torch.networks import ConcatMlp, Mlp
from adept_envs.components.robot import RobotComponentBuilder, RobotState
from adept_envs.franka.base_env import BaseFrankaEnv
from adept_envs.utils.resources import get_asset_path
from adept_envs.simulation.sim_scene import SimBackend
import pickle
ASSET_PATH = 'adept_envs/franka/assets/franka_microwave_cabinet_slider.xml'
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Class to represent a graph
class Graph:
    # A utility function to find the
    # vertex with minimum dist value, from
    # the set of vertices still in queue
    def minDistance(self, dist, queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1

        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index

    # Function to print shortest path
    # from source to j
    # using parent array
    def printPath(self, parent, j):

        # Base Case : If j is source
        if parent[j] == -1:
            print(j, end=' ')
            return
        self.printPath(parent, parent[j])
        print(j, end=' ')

    # A utility function to print
    # the constructed distance
    # array
    def printSolution(self, dist, parent):
        src = 0
        print("Vertex \t\tDistance from Source\tPath")
        for i in range(1, len(dist)):
            print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, i, dist[i])),
            self.printPath(parent, i)

    '''Function that implements Dijkstra's single source shortest path
    algorithm for a graph represented using adjacency matrix
    representation'''

    def dijkstra(self, graph, src, excluded):

        row = len(graph)
        col = len(graph[0])

        # The output array. dist[i] will hold
        # the shortest distance from src to i
        # Initialize all distances as INFINITE
        dist = [float("Inf")] * row

        # Parent array to store
        # shortest path tree
        parent = [-1] * row

        # Distance of source vertex
        # from itself is always 0
        dist[src] = 0

        # Add all vertices in queue
        queue = []
        for i in range(row):
            if excluded[i] is not None:
                queue.append(i)

        # Find shortest path for all vertices
        while queue:

            # Pick the minimum dist vertex
            # from the set of vertices
            # still in queue
            u = self.minDistance(dist, queue)

            # remove min element
            queue.remove(u)

            # Update dist value and parent
            # index of the adjacent vertices of
            # the picked vertex. Consider only
            # those vertices which are still in
            # queue
            for i in range(col):
                '''Update dist[i] only if it is in queue, there is
                an edge from u to i, and total weight of path from
                src to i through u is smaller than current value of
                dist[i]'''
                if graph[u][i] and i in queue:
                    if dist[u] + graph[u][i] < dist[i]:
                        dist[i] = dist[u] + graph[u][i]
                        parent[i] = u
        goal_paths = [[] for _ in range(len(dist))]
        for i in range(len(dist)):
            if excluded[i] is not None:
                curr_path = []
                curr_idx = i
                while parent[curr_idx] != -1:
                    curr_path = [curr_idx] + curr_path
                    curr_idx = parent[curr_idx]
                curr_path = [src] + curr_path
                goal_paths[i] = curr_path
        return goal_paths


DEFAULT_OBSERVATION_KEYS = (
    'qp',
    'obj_qp',
    'mocap_pos',
    # 'mocap_quat',
    'goal'
)
import sys
sys.path.append(".")

class Franka2Element(BaseFrankaEnv):

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
                 attempt_limit=5,
                 eval_mode=False,
                 reset_frequency=1,
                 learned_model=False,
                 learned_model_path=None,
                 counts_enabled=False,
                 graph_search=False,
                 smoothing_factor=1000,
                 idx_completion=False,
                 random_baseline=False,
                 reset_controller=False,
                 reset_objs = True,
                 **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
        """
        self._eval_mode = eval_mode
        self.reset_controller = reset_controller
        self.reset_objs = reset_objs
        self.random_baseline = random_baseline
        self.idx_completion = idx_completion
        self._reset_counter = 0
        self._reset_frequency = reset_frequency
        self._current_idx = 1
        self._goal_idx = 1
        self._counts_enabled = counts_enabled
        self.attempt_limit = attempt_limit
        self.attempt_counter = 0
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            camera_settings=camera_settings,
            sim_backend=SimBackend.DM_CONTROL,
            **kwargs)
        self.goal = np.zeros(10)
        self.use_raw_actions = use_raw_actions
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = self.sim.model.key_qvel[0].copy()
        # self.labeled_goals = pickle.load(open('sim_slider_cabinet_labeled_goals.pkl', 'rb'))
        # self.adjacency_matrix = pickle.load(open('sim_slider_cabinet_adjacency_matrix.pkl', 'rb'))
        # self._counts = np.zeros(self.adjacency_matrix.shape[0])
        self.midpoint_pos = np.array([-0.440, 0.152, 2.226])
        self.range = np.array([0.035, 0.035, 0.02])

        self.mocap_pos_clip_lower = np.array([-0.85, 0., 1.8])
        self.mocap_pos_clip_upper = np.array([0.55, 0.5, 2.7])

        self.goal_matrix = np.array([[0, 1, 1, 0],
                                     [1, 0, 0, 1],  
                                     [1, 0, 0, 1],  
                                     [0, 1, 1, 0]]) #pickle.load(open('./goal_matrix_2elements.pkl', 'rb'))
        
        
        self.smoothing_factor = smoothing_factor
        self.transition_prob = self.goal_matrix.copy()

        self._current_state = 0
        self._counts = np.zeros(4,)
        
        gpath = os.path.join(path_orig, 'end_states_2elements.pkl')
        gvpath = os.path.join(path_orig, 'end_states_2elements_vals.pkl')
        self._goals = pickle.load(open(gpath, 'rb'))
        self._goals_val = pickle.load(open(gvpath, 'rb'))

        # self._goals = pickle.load(open('./end_states_2elements_PLAYSTYLE_June14.pkl', 'rb'))
        # self._goals_val = pickle.load(open('./end_states_val_2elements_PLAYSTYLE_June14.pkl', 'rb'))

        self.eps = 0.4
        # self.smoothing_factor = smoothing_factor
        # self.transition_prob = self.goal_matrix.copy()
        self.density = np.zeros((4,))
        self.transition_prob *= self.smoothing_factor
        self.edge_visitation = self.transition_prob.copy()

        self.goal = self._goals[self._goal_idx]
        self.commanded_start = -1
        self.commanded_goal = -1
        self.learned_model = learned_model
        self.learned_model_path = learned_model_path
        self.model = None
        if self.learned_model:
            self.model = Mlp(input_size=5,
                            output_size=4,
                            hidden_sizes=(256, 256, 256))
            dat = torch.load(self.learned_model_path)
            state_dict = dat.state_dict()
            self.model.load_state_dict(state_dict)
            print("LOADED IN MODEL SUCCESSFULLY")

        self.measurement_reached_tasks = np.zeros((4, 4))
        self.measurement_commanded_tasks = np.zeros((4, 4))

        self.g = Graph()
        self.graph_search = graph_search

    def reach_particular_goal(self, goal_idx):
        assert not (self.graph_search and self.learned_model), "BOTH graph search and learned model together"
        if self.graph_search:
            print("Doing graph search eval")
            current_state = self.check_goal_completion(self.get_obs_dict()['obj_qp'])
            goal_paths = self.compute_path(current_state)
            return goal_paths[goal_idx][1]
        elif self.learned_model:
            print("learned model eval")
            o = self._get_obs(self.get_obs_dict())[2:6]
            next_task = goal_idx
            o = np.concatenate([o, [next_task]])
            input_x = torch.Tensor(o)[None, :]
            output_x = torch.nn.Softmax()(self.model(input_x)).detach().numpy()[0]
            goal_selected = np.random.choice(range(4), p=output_x)
            # print("High level BC selection for Eval " + str(output_x))
            return goal_selected
        elif self.random_baseline:
            print("random baseline eval")
            goal_selected = np.random.choice(range(4))
            return goal_selected
        else:
            print("viable goals eval")
            curr_idx = self.check_goal_completion(self.get_obs_dict()['obj_qp'])
            viable_goals = np.where(self.goal_matrix[curr_idx] > 0)[0]
            goal_idx = np.random.choice(viable_goals)
            return goal_idx

    def learned_goal_select(self, goal_selected):
        if self.learned_model:
            # print("IN LEARNED MODEL")
            o = self._get_obs(self.get_obs_dict())[2:6]
            next_task = np.random.randint(4)
            o = np.concatenate([o, [next_task]])
            input_x = torch.Tensor(o)[None, :]
            output_x = torch.nn.Softmax()(self.model(input_x)).detach().numpy()[0]
            goal_selected = np.random.choice(range(4), p=output_x)
            # print("LEARNED LIKELIHOOD PREDICTIONS " + str(output_x))

        # Updating counts
        curr_count = np.zeros((4,))
        curr_count[goal_selected] += 1
        self.update_counts(curr_count)

        if self.graph_search:
            # print("GRAPH SEARCH")
            gs, gpl, gp = self.select_goal()
            if len(gpl) > 1:
                goal_selected = gpl[1]
            else:
                goal_selected = gpl[0]

        if self.random_baseline:
            goal_selected = np.random.choice(range(4))
            if self.reset_controller:
                if self._reset_counter % 2 == 0:
                    goal_selected = 0

        return goal_selected

    def update_counts(self, new_counts):
        if self._counts_enabled:
            self._counts += new_counts

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

    def reset_arm_state(self, start_idx):
        self.last_action = None
        self.sim.reset()
        self.sim.forward()
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

        new_qpos = self.init_qpos.copy()
        new_qpos[:7] = np.array([-2.64311209, -1.76372997, -0.23182923, -2.1470029 ,  2.55216266, -0.44102682, -0.01343831])
        new_qpos[7:9] = np.array([0.1, 0.1])
        new_qpos[9:13] = self._goals_val[start_idx][2:6] # Reset to a particular state

        self.sim.data.qpos[:] = new_qpos.copy()
        self.sim.data.mocap_pos[:] = self._goals_val[start_idx][6:9] #np.array([-0.16922002,  0.07353752,  2.57067996])
        for _ in range(100):
            self.sim.step()
            self.robot.step({
                'gripper': 1 * np.ones(2)
            }, True)

    def reset_arm_only(self):
        new_qpos = self.sim.data.qpos.copy()
        self.last_action = None
        self.sim.reset()
        self.sim.forward()
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
        new_qpos[:7] = np.array(
            [-2.64311209, -1.76372997, -0.23182923, -2.1470029, 2.55216266, -0.44102682, -0.01343831])
        new_qpos[7:9] = np.array([0.1, 0.1])

        self.sim.data.qpos[:] = new_qpos.copy()
        self.sim.data.mocap_pos[:] = self._goals_val[self._goal_idx][6:9] #np.array([-0.16922002,  0.07353752,  2.57067996])
        for _ in range(100):
            self.sim.step()
            self.robot.step({
                'gripper': 1 * np.ones(2)
            }, True)

    def reset(self):
        """Resets the environment."""
        current_end_state = self.check_goal_completion(self.get_obs_dict()['obj_qp'])
        self.measurement_reached_tasks[self._current_idx][current_end_state] += 1

        if (self._reset_counter > 0) and self.graph_search and (not self._eval_mode):
            current_end_state = self.check_goal_completion(self.get_obs_dict()['obj_qp'])
            # self.update_graph(self._current_idx, current_end_state) # Current idx is the state at the start of the epiisode,
            #                                                         # current end state i that the end of the episode
            #                                                         # self._goal_idx is the state commanded during the episode
            self.update_transition_prob(init_state=self._current_state,
                                        end_state=current_end_state,
                                        commanded_goal_state=self._goal_idx)
        self.update_densities(current_end_state)
        # print("TRANSITION PROBS " + str(self.transition_prob))
        # print("DENSITY " + str(self.density))

        if self._reset_counter == 0 or self._eval_mode \
                or (self._reset_frequency != -1 and self._reset_counter % self._reset_frequency == 0):
            """ Resets the environment. """
            # print("RESETTING FULL ENVIRONMENT:")

            # ---------
            # TODO: Deal with the case with no density
            if self.commanded_start == -1:
                start_idx = np.random.randint(4)
            else:
                start_idx = self.commanded_start
            # print("START IDX" + str(start_idx))
            viable_goals = np.where(self.goal_matrix[start_idx] > 0)[0]
            if self.commanded_goal == -1:
                goal_idx = np.random.choice(viable_goals)
            else:
                goal_idx = self.commanded_goal
            # ---------

            if self.reset_objs:
                self.reset_arm_state(start_idx)
            else:
                self.reset_arm_only()

            self._goal_idx = goal_idx
            self._current_state = -1
            self.attempt_counter = 0
        else:
            print("DOING a reset free skip:")

            self.reset_arm_only()

            # More complex goal switching mechanism
            current_state = self.check_goal_completion(self.get_obs_dict()['obj_qp'])

            # If none, keep commanding the last one
            # If state is changed, set current state to new and pick the next state to go to
            if (current_state != -1 and current_state != self._current_state) or \
                    (current_state != -1 and self.attempt_counter > self.attempt_limit):
                viable_goals = np.where(self.goal_matrix[current_state] > 0)[0]
                goal_selected = np.random.choice(viable_goals)
                goal_selected = self.learned_goal_select(goal_selected)
                self._current_state = current_state
                self._goal_idx = goal_selected
                self._current_state = current_state
                self.attempt_counter = 0

            # If state is not changed, with some eps likelihood, change which goal is commanded
            elif current_state != -1 and current_state == self._current_state:
                self.attempt_counter += 1
                if np.random.rand() < self.eps:
                    print("Selecting a new goal, tried too many times")
                    viable_goals = np.where(self.goal_matrix[current_state] > 0)[0]
                    goal_selected = np.random.choice(viable_goals)
                    goal_selected = self.learned_goal_select(goal_selected)
                    self._goal_idx = goal_selected
                    self.attempt_counter = 0

        self._reset_counter += 1
        # print("EVAL MODE IS " + str(self._eval_mode))
        # print("Current goal is %d" % self._goal_idx)
        self._current_idx = self.check_goal_completion(self.get_obs_dict()['obj_qp'][None, :].squeeze(axis=0))
        self.goal = self._goals[self._goal_idx]
        # self.goal_idx = self._goal_idx
        # TODO: current idx and current_stte are a bit repeate
        obs_dict = self.get_obs_dict()

        self.measurement_commanded_tasks[self._current_idx][self._goal_idx] += 1
        return self._get_obs(obs_dict)

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(5,))


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

    def compute_path(self, current_state):
        goal_paths = self.g.dijkstra(self.goal_matrix, current_state, self._goals)
        return goal_paths

    def update_graph(self, init_state, goal_state):
        self.goal_matrix[init_state][goal_state] = 1

    def update_transition_prob(self, init_state, end_state, commanded_goal_state):
        if end_state == commanded_goal_state:
            self.transition_prob[init_state][commanded_goal_state] += 1.
        else:
            self.transition_prob[init_state][commanded_goal_state] += 0.

        self.edge_visitation[init_state][commanded_goal_state] += 1.

    def update_densities(self, goal_state):
        self.density[goal_state] += 1

    def select_goal(self):
        current_state = self.check_goal_completion(self.get_obs_dict()['obj_qp'])
        goal_paths = self.compute_path(current_state)
        density_diffs = [-np.inf for i in range(4)]
        # TODO: Deal with the cases with no path?
        for i in range(4):
            if i == current_state:
                density_diffs[i] = -np.inf
                continue

            if self._goals[i] is None:
                continue

            added_density = np.zeros(4)
            weighted_prob = 1
            curr_path_idx = current_state
            for gs in goal_paths[i][1:]:
                weighted_prob *= (self.transition_prob[curr_path_idx][gs]/self.edge_visitation[curr_path_idx][gs])
                added_density[gs] = weighted_prob
                curr_path_idx = gs
            new_density = self.density + added_density
            new_density = new_density / new_density.sum()

            # TVD to the uniform
            density_diffs[i] = -np.sum(np.abs(new_density - np.ones(4) * (1 / 4.)))

        new_goal = np.argmax(np.array(density_diffs))
        return new_goal, goal_paths[new_goal], goal_paths

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
            camera = engine.MovableCamera(self.sim, 480, 640)
            camera.set_pose(
                distance=2.2, lookat=[-0.2, .5, 2.1], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            super().render()

    def check_goal_completion(self, curr_pos):
        max_objs = np.array([0.17, 1, 0.6, -0.05])
        min_objs = np.array([0.08, 0.1, 0.2, -0.2])
        init_bitflips = np.array([0, 0, 0, 1])
        curr_bitflips = init_bitflips.copy()
        if len(curr_pos.shape) > 1:
            curr_pos = curr_pos.squeeze(axis=0)
        for j in range(4):
            if curr_pos[j] > max_objs[j]:
                curr_bitflips[j] = 1
            elif curr_pos[j] < min_objs[j]:
                curr_bitflips[j] = 0
        new_idx = 2 * curr_bitflips[0] + curr_bitflips[2]
        return new_idx

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if self.idx_completion:
            """Returns a standardized measure of success for the environment."""
            curr_idx = self.check_goal_completion(obs_dict['obj_qp'])
            rd = collections.OrderedDict((
                ('ee_slider', np.array([np.float(curr_idx == self._goal_idx)])),
            ))
        else:
            self.slider_arm_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('slide_site')]
            self.cabinet_arm_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('hinge_site2')]
            
            self.slider_open_pos = self._goals_val[self._goal_idx][2]
            self.cabinet_open_pos = self._goals_val[self._goal_idx][4]
            
            self.slider_closed_pos = self._goals_val[self._goal_idx][2]
            self.cabinet_closed_pos = self._goals_val[self._goal_idx][4]
            
            goal_idx = np.binary_repr(self._goal_idx)
            curr_idx = np.binary_repr(self._current_idx)
            goal_idx = '0'*(2 - len(goal_idx)) + goal_idx
            curr_idx = '0'*(2 - len(curr_idx)) + curr_idx
            assert len(goal_idx) == len(curr_idx)
            i = 0
            for i in range(len(curr_idx)):
                if goal_idx[i] != curr_idx[i]:
                    break

            if goal_idx[i] == '1':
                goal_open = True
            else:
                goal_open = False
            normalized_dist = [0.22874171, 1.01982685]
            open_arm_positions = [self.slider_arm_pos,
                                  self.cabinet_arm_pos]
            closed_arm_positions = [self.slider_arm_pos,
                                  self.cabinet_arm_pos]
            open_element_positions = [0.4,
                                      self.cabinet_open_pos]
            closed_element_positions = [self.slider_closed_pos,
                                      self.cabinet_closed_pos]

            if goal_open:
                arm_target_pos = closed_arm_positions[i]
                element_target_pos = open_element_positions[i]
            else:
                arm_target_pos = open_arm_positions[i]
                element_target_pos = closed_element_positions[i]

            if i == 0:
                obj_pos = obs_dict['obj_qp'][:, 0]
            else:
                obj_pos = obs_dict['obj_qp'][:, 2]
            arm_error = obs_dict['mocap_pos'] - arm_target_pos
            slider_error = (obj_pos - element_target_pos)
            normalizer = normalized_dist[i]
            rd = collections.OrderedDict((
                ('ee_slider', np.array([-20*np.float(np.linalg.norm(slider_error))/normalizer])),
                ('arm_dist', np.array([-np.float(np.linalg.norm(arm_error))])),
            ))

        return rd