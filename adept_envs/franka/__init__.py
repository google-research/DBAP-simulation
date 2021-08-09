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

"""Gym environment registration for Franka environments."""

from adept_envs.utils.registration import register

#===============================================================================
# Orient tasks
#===============================================================================

# Default number of steps per episode.
_ORIENT_EPISODE_LEN = 100  # 100*40*2.5ms = 10s

register(
    env_id='BaseFrankaOrient-v0',
    class_path='adept_envs.franka.orient:BaseFrankaOrient',
    max_episode_steps=_ORIENT_EPISODE_LEN
)

register(
    env_id='FrankaDoorOpening-v0',
    class_path='adept_envs.franka.door_opening:FrankaDoorOpening',
    max_episode_steps=_ORIENT_EPISODE_LEN
)


#===============================================================================
# Kitchen Tasks
#===============================================================================

# Relax the robot
register(
    env_id='kitchen_relax-v1',
    class_path='adept_envs.franka.kitchen_multitask:KitchenTaskRelaxV1',
    max_episode_steps=100,
)

register(
    env_id='franka_reachfree-v1',
    class_path='adept_envs.franka.franka_reach:FrankaReach',
    max_episode_steps=100,
)

register(
    env_id='franka_slide-v1',
    class_path='adept_envs.franka.franka_slide:FrankaSlide',
    max_episode_steps=100,
)


register(
    env_id='franka_slide_vr-v1',
    class_path='adept_envs.franka.franka_slide_old:FrankaSlide',
    max_episode_steps=100,
)

register(
    env_id='franka_microwave-v1',
    class_path='adept_envs.franka.franka_microwave:FrankaMicrowave',
    max_episode_steps=100,
)

register(
    env_id='franka_microwave_cabinet_slider-v1',
    class_path='adept_envs.franka.franka_microwave_cabinet_slider:FrankaMicrowaveCabinetSlider',
    max_episode_steps=100,
)

register(
    env_id='franka_microwave_cabinet_slider_resetfree-v1',
    class_path='adept_envs.franka.franka_microwave_cabinet_slider_resetfree:FrankaMicrowaveCabinetSlider',
    max_episode_steps=100,
)

register(
    env_id='franka_microwave_cabinet_slider_eval-v1',
    class_path='adept_envs.franka.franka_microwave_cabinet_slider_eval:FrankaMicrowaveCabinetSlider',
    max_episode_steps=100,
)

register(
    env_id='franka_cabinet_slider_knob_switch_toaster_resetfree-v1',
    class_path='adept_envs.franka.franka_cabinet_slider_knob_switch_toaster:FrankaCabinetSliderKnobSwitchToaster',
    max_episode_steps=100,
)

register(
    env_id='franka_cabinet_slider_resetfree_newcode-v1',
    class_path='adept_envs.franka.franka_2element_newcode:Franka2Element',
    max_episode_steps=100,
)

register(
    env_id='franka_microwave_cabinet_slider_resetfree_newcode-v1',
    class_path='adept_envs.franka.franka_3element_newcode:Franka3Element',
    max_episode_steps=100,
)

register(
    env_id='franka_knob_cabinet_slider_resetfree_newcode-v1',
    class_path='adept_envs.franka.franka_3element_knob_newcode:Franka3ElementKnob',
    max_episode_steps=100,
)




# Reach for a pose.
register(
    env_id='KitchenPoseTest-v0',
    class_path='adept_envs.franka.kitchen_multitask_vr:KitchenPoseTest',
    max_episode_steps=100,
)

# Reach for a pose using Velocity actuation.
register(
    env_id='KitchenPoseTestVelAct-v0',
    class_path='adept_envs.franka.kitchen_multitask_vr:KitchenPoseTestVelAct',
    max_episode_steps=100,
)
