# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from __future__ import annotations

try:
    from isaacgym import gymapi, gymutil
except ImportError:
    raise Exception("Please install isaacgym to run Isaac Gym tasks!")

try :
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_finger import ShadowHandCatchOver2Underarm_Safe_finger
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_joint import ShadowHandCatchOver2Underarm_Safe_joint
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_finger import ShadowHandOver_Safe_finger
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_joint import ShadowHandOver_Safe_joint

    from safepo.common.wrappers import GymnasiumIsaacEnv
except ImportError:
    pass

import argparse
import os
import json
import numpy as np
import torch
import pickle as pkl

from argparse import Namespace

import numpy as np
import torch
from gymnasium import spaces

from isaacgym import gymtorch
import torch
import numpy as np
import os
from PIL import Image


isaac_gym_map = {
    "ShadowHandOver_Safe_finger": "shadow_hand_over_safe_finger",
    "ShadowHandOver_Safe_joint": "shadow_hand_over_safe_joint",
    "ShadowHandCatchOver2Underarm_Safe_finger": "shadow_hand_catch_over_2_underarm_safe_finger",
    "ShadowHandCatchOver2Underarm_Safe_joint": "shadow_hand_catch_over_2_underarm_safe_joint",
    "FreightFrankaCloseDrawer": "freight_franka_close_drawer",
    "FreightFrankaPickAndPlace": "freight_franka_pick_and_place",
}


import json
import os
from random import randint, shuffle
from time import time

import numpy as np
import yaml
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from tqdm import tqdm

import operator
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.gymutil import (
    apply_random_samples,
    check_buckets,
    generate_random_samples,
    get_default_setter_args,
    get_property_getter_map,
    get_property_setter_map,
)

import inspect

from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.freight_franka_pick_and_place import FreightFrankaPickAndPlace as YetAnotherFreightFrankaPickAndPlace
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.freight_franka_close_drawer import FreightFrankaCloseDrawer as YetAnotherFreightFrankaCloseDrawer


# Base class for RL tasks
class BaseTask:
    def __init__(self, cfg, enable_camera_sensors=False):
        self.gym = gymapi.acquire_gym()

        self.device_type = cfg.get('device_type', 'cuda')
        self.device_id = cfg.get('device_id', 0)

        self.device = 'cpu'
        if self.device_type == 'cuda' or self.device_type == 'GPU':
            self.device = 'cuda' + ':' + str(self.device_id)

        self.headless = cfg['headless']

        # double check!
        self.graphics_device_id = self.device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg['env']['numEnvs']
        self.num_obs = cfg['env']['numObservations']
        self.num_states = cfg['env'].get('numStates', 0)
        self.num_actions = cfg['env']['numActions']

        self.control_freq_inv = cfg['env'].get('controlFrequencyInv', 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float
        )
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.cost_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # # if running with a viewer, set up keyboard shortcuts and camera
        # if self.headless == False:
        #     # subscribe to keyboard shortcuts
        #     self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        #     self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, 'QUIT')
        #     self.gym.subscribe_viewer_keyboard_event(
        #         self.viewer, gymapi.KEY_V, 'toggle_viewer_sync'
        #     )

        #     # set the camera position based on up axis
        #     sim_params = self.gym.get_sim_params(self.sim)
        #     if sim_params.up_axis == gymapi.UP_AXIS_Z:
        #         cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
        #         cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
        #     else:
        #         cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
        #         cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

        #     self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            


    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print('*** Failed to create sim')
            quit()

        return sim

    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def get_states(self):
        return self.states_buf

    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == 'QUIT' and evt.value > 0:
                    sys.exit()
                elif evt.action == 'toggle_viewer_sync' and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def get_actor_params_info(self, dr_params, env):
        """Returns a flat array of actor params, their names and ranges."""
        if 'actor_params' not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params['actor_params'].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name + '_' + str(prop_idx) + '_' + attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0 * float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name + '_' + str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get('frequency', 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(
                self.randomize_buf >= rand_freq,
                torch.ones_like(self.randomize_buf),
                torch.zeros_like(self.randomize_buf),
            )
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ['observations', 'actions']:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]['distribution']
                op_type = dr_params[nonphysical_param]['operation']
                sched_type = (
                    dr_params[nonphysical_param]['schedule']
                    if 'schedule' in dr_params[nonphysical_param]
                    else None
                )
                sched_step = (
                    dr_params[nonphysical_param]['schedule_steps']
                    if 'schedule' in dr_params[nonphysical_param]
                    else None
                )
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]['range']
                    mu_corr, var_corr = dr_params[nonphysical_param].get(
                        'range_correlated', [0.0, 0.0]
                    )

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * (
                            1.0 - sched_scaling
                        )  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * (
                            1.0 - sched_scaling
                        )  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu']
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        'mu': mu,
                        'var': var,
                        'mu_corr': mu_corr,
                        'var_corr': var_corr,
                        'noise_lambda': noise_lambda,
                    }

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]['range']
                    lo_corr, hi_corr = dr_params[nonphysical_param].get(
                        'range_correlated', [0.0, 0.0]
                    )

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(
                            tensor,
                            corr
                            + torch.rand_like(tensor) * (params['hi'] - params['lo'])
                            + params['lo'],
                        )

                    self.dr_randomizations[nonphysical_param] = {
                        'lo': lo,
                        'hi': hi,
                        'lo_corr': lo_corr,
                        'hi_corr': hi_corr,
                        'noise_lambda': noise_lambda,
                    }

        if 'sim_params' in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params['sim_params']
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props['sim_params'] = {
                    attr: getattr(prop, attr) for attr in dir(prop)
                }

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop,
                    self.original_props['sim_params'],
                    attr,
                    attr_randomization_params,
                    self.last_step,
                )

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params['actor_params'].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(
                                env,
                                handle,
                                n,
                                gymapi.MESH_VISUAL,
                                gymapi.Vec3(
                                    random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
                                ),
                            )
                        continue
                    if prop_name == 'scale':
                        attr_randomization_params = prop_attrs
                        sample = generate_random_samples(
                            attr_randomization_params, 1, self.last_step, None
                        )
                        og_scale = 1
                        if attr_randomization_params['operation'] == 'scaling':
                            new_scale = og_scale * sample
                        elif attr_randomization_params['operation'] == 'additive':
                            new_scale = og_scale + sample
                        self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop
                            ]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], p, attr
                                    )
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params, self.last_step, smpl
                                )
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            smpl = None
                            if self.actor_params_generator is not None:
                                smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                    extern_sample, extern_offsets[env_id], prop, attr
                                )
                            apply_random_samples(
                                prop,
                                self.original_props[prop_name],
                                attr,
                                attr_randomization_params,
                                self.last_step,
                                smpl,
                            )

                    setter = param_setters_map[prop_name]
                    default_args = param_setter_defaults_map[prop_name]
                    setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print(
                            'env_id',
                            env_id,
                            'extern_offset',
                            extern_offsets[env_id],
                            'vs extern_sample.shape',
                            extern_sample.shape,
                        )
                        raise Exception('Invalid extern_sample size')

        self.first_randomization = False

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset : offset + prop[attr].shape[0]]
        return smpl, offset + prop[attr].shape[0]
    else:
        return sample[offset], offset + 1


def quat_axis(q, axis=0):
    """??"""
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


class FreightFrankaCloseDrawer(BaseTask):
    def __init__(
        self,
        cfg,
        sim_params,
        physics_engine,
        device_type,
        device_id,
        headless,
        agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]],
        is_multi_agent=False,
        log_dir=None,
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.log_dir = log_dir
        self.up_axis = 'z'
        self.device_id = device_id if 'cuda' not in str(device_id) else int(device_id[-1])
        self.cfg['device_type'] = device_type
        self.cfg['device_id'] = self.device_id
        self.cfg['headless'] = headless
        self.device_type = device_type
        self.headless = headless
        self.device = 'cpu'
        self.use_handle = False
        if self.device_type == 'cuda' or self.device_type == 'GPU':
            self.device = 'cuda' + ':' + str(self.device_id)
        self.max_episode_length = self.cfg['env']['maxEpisodeLength']

        self.env_num_train = cfg['env']['numEnvs']
        self.env_num = self.env_num_train
        original_class_file_path = inspect.getfile(YetAnotherFreightFrankaCloseDrawer)
        
        self.asset_root = os.path.dirname(original_class_file_path).replace(
            'envs/tasks', 'envs/assets'
        )
        self.cabinet_num_train = cfg['env']['asset']['cabinetAssetNumTrain']
        self.cabinet_num = self.cabinet_num_train
        cabinet_train_list_len = len(cfg['env']['asset']['trainAssets'])
        self.cabinet_train_name_list = []
        self.exp_name = cfg['env']['env_name']
        print('Simulator: number of cabinets', self.cabinet_num)
        print('Simulator: number of environments', self.env_num)
        if self.cabinet_num_train:
            assert self.env_num_train % self.cabinet_num_train == 0

        assert (
            self.cabinet_num_train <= cabinet_train_list_len
        )  # the number of used length must less than real length
        assert self.env_num % self.cabinet_num == 0  # each cabinet should have equal number envs
        self.env_per_cabinet = self.env_num // self.cabinet_num
        self.task_meta = {
            'training_env_num': self.env_num_train,
            'need_update': True,
            'max_episode_length': self.max_episode_length,
            'obs_dim': cfg['env']['numObservations'],
        }
        for name in cfg['env']['asset']['trainAssets']:
            self.cabinet_train_name_list.append(cfg['env']['asset']['trainAssets'][name]['name'])

        self.cabinet_dof_lower_limits_tensor = torch.zeros(
            (self.cabinet_num, 1), device=self.device
        )
        self.cabinet_dof_upper_limits_tensor = torch.zeros(
            (self.cabinet_num, 1), device=self.device
        )
        self.cabinet_handle_pos_tensor = torch.zeros((self.cabinet_num, 3), device=self.device)
        self.cabinet_have_handle_tensor = torch.zeros((self.cabinet_num,), device=self.device)
        self.cabinet_open_dir_tensor = torch.zeros((self.cabinet_num,), device=self.device)
        self.cabinet_door_min_tensor = torch.zeros((self.cabinet_num, 3), device=self.device)
        self.cabinet_door_max_tensor = torch.zeros((self.cabinet_num, 3), device=self.device)

        self.env_ptr_list = []
        self.obj_loaded = False
        self.franka_loaded = False

        self.use_handle = cfg['task']['useHandle']
        self.use_stage = cfg['task']['useStage']
        self.use_slider = cfg['task']['useSlider']

        if 'useTaskId' in self.cfg['task'] and self.cfg['task']['useTaskId']:
            self.cfg['env']['numObservations'] += self.cabinet_num

        super().__init__(cfg=self.cfg, enable_camera_sensors=cfg['env']['enableCameraSensors'])
        # acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_tensor = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        if (
            self.cfg['env']['driveMode'] == 'ik'
        ):  # inverse kinetic needs jacobian tensor, other drive mode don't need
            self.jacobian_tensor = gymtorch.wrap_tensor(
                self.gym.acquire_jacobian_tensor(self.sim, 'franka')
            )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_tensor = self.root_tensor.view(self.num_envs, -1, 13)
        self.dof_state_tensor = self.dof_state_tensor.view(self.num_envs, -1, 2)
        self.rigid_body_tensor = self.rigid_body_tensor.view(self.num_envs, -1, 13)
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        self.initial_dof_states = self.dof_state_tensor.clone()
        self.initial_root_states = self.root_tensor.clone()
        self.initial_rigid_body_states = self.rigid_body_tensor.clone()

        # precise slices of tensors
        env_ptr = self.env_ptr_list[0]
        franka1_actor = self.franka_actor_list[0]
        cabinet_actor = self.cabinet_actor_list[0]
        self.hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'panda_hand', gymapi.DOMAIN_ENV
        )
        self.freight_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'base_link', gymapi.DOMAIN_ENV
        )
        self.hand_lfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'panda_leftfinger', gymapi.DOMAIN_ENV
        )
        self.hand_rfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, franka1_actor, 'panda_rightfinger', gymapi.DOMAIN_ENV
        )
        self.cabinet_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, cabinet_actor, self.cabinet_rig_name, gymapi.DOMAIN_ENV
        )
        self.cabinet_base_rigid_body_index = self.gym.find_actor_rigid_body_index(
            env_ptr, cabinet_actor, self.cabinet_base_rig_name, gymapi.DOMAIN_ENV
        )
        self.cabinet_dof_index = self.gym.find_actor_dof_index(
            env_ptr, cabinet_actor, self.cabinet_dof_name, gymapi.DOMAIN_ENV
        )

        self.hand_rigid_body_tensor = self.rigid_body_tensor[:, self.hand_rigid_body_index, :]
        self.franka_dof_tensor = self.dof_state_tensor[:, : self.franka_num_dofs, :]
        self.cabinet_dof_tensor = self.dof_state_tensor[:, self.cabinet_dof_index, :]
        self.cabinet_dof_tensor_spec = self._detailed_view(self.cabinet_dof_tensor)
        self.cabinet_door_rigid_body_tensor = self.rigid_body_tensor[
            :, self.cabinet_rigid_body_index, :
        ]
        self.franka_root_tensor = self.root_tensor[:, 0, :]
        self.cabinet_root_tensor = self.root_tensor[:, 1, :]

        self.cabinet_dof_target = self.initial_dof_states[:, self.cabinet_dof_index, 0]
        self.dof_dim = self.franka_num_dofs + 1
        self.pos_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.stage = torch.zeros((self.num_envs), device=self.device)

        # initialization of pose
        self.cabinet_dof_coef = -1.0
        self.success_dof_states = self.cabinet_dof_lower_limits_tensor[:, 0].clone()
        self.initial_dof_states.view(self.cabinet_num, self.env_per_cabinet, -1, 2)[
            :, :, self.cabinet_dof_index, 0
        ] = (torch.ones((self.cabinet_num, 1), device=self.device) * 0.2)

        self.map_dis_bar = cfg['env']['map_dis_bar']
        self.action_speed_scale = cfg['env']['actionSpeedScale']

        # params of randomization
        self.cabinet_reset_position_noise = cfg['env']['reset']['cabinet']['resetPositionNoise']
        self.cabinet_reset_rotation_noise = cfg['env']['reset']['cabinet']['resetRotationNoise']
        self.cabinet_reset_dof_pos_interval = cfg['env']['reset']['cabinet'][
            'resetDofPosRandomInterval'
        ]
        self.cabinet_reset_dof_vel_interval = cfg['env']['reset']['cabinet'][
            'resetDofVelRandomInterval'
        ]
        self.franka_reset_position_noise = cfg['env']['reset']['franka']['resetPositionNoise']
        self.franka_reset_rotation_noise = cfg['env']['reset']['franka']['resetRotationNoise']
        self.franka_reset_dof_pos_interval = cfg['env']['reset']['franka'][
            'resetDofPosRandomInterval'
        ]
        self.franka_reset_dof_vel_interval = cfg['env']['reset']['franka'][
            'resetDofVelRandomInterval'
        ]

        # params for success rate
        self.success = torch.zeros((self.env_num,), device=self.device)
        self.success_rate = torch.zeros((self.env_num,), device=self.device)
        self.success_buf = torch.zeros((self.env_num,), device=self.device).long()

        self.average_reward = None

        # flags for switching between training and evaluation mode
        self.train_mode = True

        self.num_freight_obs = 3 * 2
        self.num_franka_obs = 9 * 2
        
        # Initialize camera properties
        camera_props = gymapi.CameraProperties()
        camera_props.width = 512
        camera_props.height = 512
        camera_props.enable_tensors = True  # Enable tensor access

        self.cameras = []

        # Create cameras in each environment
        for env in tqdm(self.env_ptr_list, desc='Creating cameras'):
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            # import pdb; pdb.set_trace()
            # Set camera position and orientation
            camera_position = gymapi.Vec3(2.0, 2.0, 2.0)  # Adjust as needed
            camera_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.set_camera_location(camera_handle, env, camera_position, camera_target)

            self.cameras.append(camera_handle)
        
        # Initialize frame indices and episode counters for each environment
        self.env_frame_idx = torch.zeros(self.env_num, dtype=torch.long, device=self.device)
        self.env_episode_idx = torch.zeros(self.env_num, dtype=torch.long, device=self.device)

        self.output_base_dir = 'output_images'
        self.output_dirs = []

        for env_id in tqdm(range(self.env_num), desc='Creating output directories'):
            env_dir = os.path.join(self.output_base_dir, f'env_{env_id}')
            os.makedirs(env_dir, exist_ok=True)
            self.output_dirs.append(env_dir)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        self._create_ground_plane()
        self._place_agents(self.cfg['env']['numEnvs'], self.cfg['env']['envSpacing'])

    def _franka_init_pose(self):
        initial_franka_pose = gymapi.Transform()

        initial_franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        if self.cfg['task']['target'] == 'close':
            self.cabinet_dof_coef = -1.0
            self.success_dof_states = self.cabinet_dof_lower_limits_tensor[:, 0].clone()
            initial_franka_pose.p = gymapi.Vec3(0.6, 0.0, 0.02)

        return initial_franka_pose

    def _cam_pose(self):
        cam_pos = gymapi.Vec3(13.0, 13.0, 6.0)
        cam_target = gymapi.Vec3(8.0, 8.0, 0.1)

        return cam_pos, cam_target

    def _load_franka(self, env_ptr, env_id):
        if self.franka_loaded == False:
            self.franka_actor_list = []

            asset_root = self.asset_root
            asset_file = 'urdf/freight_franka/urdf/freight_franka.urdf'
            self.gripper_length = 0.13
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
            asset_options.flip_visual_attachments = True
            asset_options.armature = 0.01
            self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            self.franka_loaded = True

        (
            franka_dof_max_torque,
            franka_dof_lower_limits,
            franka_dof_upper_limits,
        ) = self._get_dof_property(self.franka_asset)
        self.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=self.device)
        self.franka_dof_mean_limits_tensor = torch.tensor(
            (franka_dof_lower_limits + franka_dof_upper_limits) / 2, device=self.device
        )
        self.franka_dof_limits_range_tensor = torch.tensor(
            (franka_dof_upper_limits - franka_dof_lower_limits) / 2, device=self.device
        )
        self.franka_dof_lower_limits_tensor = torch.tensor(
            franka_dof_lower_limits, device=self.device
        )
        self.franka_dof_upper_limits_tensor = torch.tensor(
            franka_dof_upper_limits, device=self.device
        )

        dof_props = self.gym.get_asset_dof_properties(self.franka_asset)

        # use position drive for all dofs
        dof_props['driveMode'][:-2].fill(gymapi.DOF_MODE_POS)
        dof_props['stiffness'][:-2].fill(400.0)
        dof_props['damping'][:-2].fill(40.0)
        # grippers
        dof_props['driveMode'][-2:].fill(gymapi.DOF_MODE_EFFORT)
        dof_props['stiffness'][-2:].fill(0.0)
        dof_props['damping'][-2:].fill(0.0)

        # root pose
        initial_franka_pose = self._franka_init_pose()

        # set start dof
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.3
        # grippers open
        default_dof_pos[-2:] = franka_dof_upper_limits[-2:]
        franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype)
        franka_dof_state['pos'] = default_dof_pos

        franka_actor = self.gym.create_actor(
            env_ptr, self.franka_asset, initial_franka_pose, 'franka', env_id, 1, 0
        )

        self.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
        self.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
        self.franka_actor_list.append(franka_actor)

    def _get_dof_property(self, asset):
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        for i in range(dof_num):
            dof_max_torque.append(dof_props['effort'][i])
            dof_lower_limits.append(dof_props['lower'][i])
            dof_upper_limits.append(dof_props['upper'][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits

    def _obj_init_pose(self, min_dict, max_dict):
        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(-max_dict[2], 0.0, -min_dict[1] + 0.3)
        cabinet_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        return cabinet_start_pose

    def _load_obj_asset(self):
        self.cabinet_asset_name_list = []
        self.cabinet_asset_list = []
        self.cabinet_pose_list = []
        self.cabinet_actor_list = []
        self.cabinet_pc = []

        train_len = len(self.cfg['env']['asset']['trainAssets'].items())
        train_len = min(train_len, self.cabinet_num_train)
        total_len = train_len
        used_len = min(total_len, self.cabinet_num)

        random_asset = self.cfg['env']['asset']['randomAsset']
        select_train_asset = [i for i in range(train_len)]
        if (
            random_asset
        ):  # if we need random asset list from given dataset, we shuffle the list to be read
            shuffle(select_train_asset)
        select_train_asset = select_train_asset[:train_len]

        with tqdm(total=used_len) as pbar:
            pbar.set_description('Loading cabinet assets:')
            cur = 0

            asset_list = []

            # prepare the assets to be used
            for id, (name, val) in enumerate(self.cfg['env']['asset']['trainAssets'].items()):
                if id in select_train_asset:
                    asset_list.append((id, (name, val)))

            for id, (name, val) in asset_list:
                self.cabinet_asset_name_list.append(name)

                asset_options = gymapi.AssetOptions()
                asset_options.fix_base_link = True
                asset_options.disable_gravity = True
                asset_options.collapse_fixed_joints = True
                asset_options.use_mesh_materials = True
                asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                asset_options.override_com = True
                asset_options.override_inertia = True
                asset_options.vhacd_enabled = True
                asset_options.vhacd_params = gymapi.VhacdParams()
                asset_options.vhacd_params.resolution = 512

                cabinet_asset = self.gym.load_asset(
                    self.sim, self.asset_root, val['path'], asset_options
                )
                self.cabinet_asset_list.append(cabinet_asset)

                with open(os.path.join(self.asset_root, val['boundingBox'])) as f:
                    cabinet_bounding_box = json.load(f)
                    min_dict = cabinet_bounding_box['min']
                    max_dict = cabinet_bounding_box['max']

                dof_dict = self.gym.get_asset_dof_dict(cabinet_asset)
                if len(dof_dict) != 1:
                    print(val['path'])
                    print(len(dof_dict))
                assert len(dof_dict) == 1
                self.cabinet_dof_name = list(dof_dict.keys())[0]

                rig_dict = self.gym.get_asset_rigid_body_dict(cabinet_asset)
                assert len(rig_dict) == 2
                self.cabinet_rig_name = list(rig_dict.keys())[1]
                self.cabinet_base_rig_name = list(rig_dict.keys())[0]
                assert self.cabinet_rig_name != 'base'

                self.cabinet_pose_list.append(self._obj_init_pose(min_dict, max_dict))

                max_torque, lower_limits, upper_limits = self._get_dof_property(cabinet_asset)
                self.cabinet_dof_lower_limits_tensor[cur, :] = torch.tensor(
                    lower_limits[0], device=self.device
                )
                self.cabinet_dof_upper_limits_tensor[cur, :] = torch.tensor(
                    upper_limits[0], device=self.device
                )

                dataset_path = self.cfg['env']['asset']['datasetPath']

                with open(os.path.join(self.asset_root, dataset_path, name, 'handle.yaml')) as f:
                    handle_dict = yaml.safe_load(f)
                    self.cabinet_have_handle_tensor[cur] = handle_dict['has_handle']
                    self.cabinet_handle_pos_tensor[cur][0] = handle_dict['pos']['x']
                    self.cabinet_handle_pos_tensor[cur][1] = handle_dict['pos']['y']
                    self.cabinet_handle_pos_tensor[cur][2] = handle_dict['pos']['z']

                with open(os.path.join(self.asset_root, dataset_path, name, 'door.yaml')) as f:
                    door_dict = yaml.safe_load(f)
                    if (door_dict['open_dir'] not in [-1, 1]) and (
                        not self.cfg['task']['useDrawer']
                    ):
                        print(
                            'Warning: Door type of {} is not supported, possibly a unrecognized open direction',
                            name,
                        )
                    if self.cfg['task']['useDrawer']:
                        self.cabinet_open_dir_tensor[cur] = 1
                    else:
                        self.cabinet_open_dir_tensor[cur] = door_dict['open_dir']
                    self.cabinet_door_min_tensor[cur][0] = door_dict['bounding_box']['xmin']
                    self.cabinet_door_min_tensor[cur][1] = door_dict['bounding_box']['ymin']
                    self.cabinet_door_min_tensor[cur][2] = door_dict['bounding_box']['zmin']
                    self.cabinet_door_max_tensor[cur][0] = door_dict['bounding_box']['xmax']
                    self.cabinet_door_max_tensor[cur][1] = door_dict['bounding_box']['ymax']
                    self.cabinet_door_max_tensor[cur][2] = door_dict['bounding_box']['zmax']

                pbar.update(1)
                cur += 1
        # flag for hazard which is forbidden to enter
        hazard_asset_options = gymapi.AssetOptions()
        hazard_asset_options.fix_base_link = True
        hazard_asset_options.disable_gravity = False
        self.hazard_asset = self.gym.create_box(self.sim, 0.5, 1.0, 0.01, hazard_asset_options)
        self.hazard_pose = gymapi.Transform()
        self.hazard_pose.p = gymapi.Vec3(0.05, 0.0, 0.005)  # franka:0, 0.0, 0
        self.hazard_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    def _load_obj(self, env_ptr, env_id):
        if self.obj_loaded == False:
            self._load_obj_asset()

            self.cabinet_handle_pos_tensor = self.cabinet_handle_pos_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_have_handle_tensor = self.cabinet_have_handle_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_open_dir_tensor = self.cabinet_open_dir_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_door_min_tensor = self.cabinet_door_min_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )
            self.cabinet_door_max_tensor = self.cabinet_door_max_tensor.repeat_interleave(
                self.env_per_cabinet, dim=0
            )

            self.cabinet_door_edge_min_l = torch.zeros_like(self.cabinet_door_min_tensor)
            self.cabinet_door_edge_max_l = torch.zeros_like(self.cabinet_door_max_tensor)
            self.cabinet_door_edge_min_r = torch.zeros_like(self.cabinet_door_min_tensor)
            self.cabinet_door_edge_max_r = torch.zeros_like(self.cabinet_door_max_tensor)
            self.cabinet_door_edge_min = torch.zeros_like(self.cabinet_door_min_tensor)
            self.cabinet_door_edge_max = torch.zeros_like(self.cabinet_door_max_tensor)
            self.cabinet_door_edge_min_l[:, 0] = self.cabinet_door_max_tensor[:, 0]
            self.cabinet_door_edge_min_l[:, 1] = self.cabinet_door_min_tensor[:, 1]
            self.cabinet_door_edge_min_l[:, 2] = self.cabinet_door_min_tensor[:, 2]
            self.cabinet_door_edge_max_l[:, 0] = self.cabinet_door_max_tensor[:, 0]
            self.cabinet_door_edge_max_l[:, 1] = self.cabinet_door_max_tensor[:, 1]
            self.cabinet_door_edge_max_l[:, 2] = self.cabinet_door_min_tensor[:, 2]

            self.cabinet_door_edge_min_r[:, 0] = self.cabinet_door_min_tensor[:, 0]
            self.cabinet_door_edge_min_r[:, 1] = self.cabinet_door_min_tensor[:, 1]
            self.cabinet_door_edge_min_r[:, 2] = self.cabinet_door_min_tensor[:, 2]
            self.cabinet_door_edge_max_r[:, 0] = self.cabinet_door_min_tensor[:, 0]
            self.cabinet_door_edge_max_r[:, 1] = self.cabinet_door_max_tensor[:, 1]
            self.cabinet_door_edge_max_r[:, 2] = self.cabinet_door_min_tensor[:, 2]
            self.cabinet_door_edge_max = torch.where(
                self.cabinet_open_dir_tensor.view(self.num_envs, -1) < -0.5,
                self.cabinet_door_edge_max_l,
                self.cabinet_door_edge_max_r,
            )
            self.cabinet_door_edge_min = torch.where(
                self.cabinet_open_dir_tensor.view(self.num_envs, -1) < -0.5,
                self.cabinet_door_edge_min_l,
                self.cabinet_door_edge_min_r,
            )

            self.obj_loaded = True

        obj_actor = self.gym.create_actor(
            env_ptr,
            self.cabinet_asset_list[0],
            self.cabinet_pose_list[0],
            f'cabinet{env_id}',
            env_id,
            2,
            0,
        )
        cabinet_dof_props = self.gym.get_asset_dof_properties(self.cabinet_asset_list[0])
        cabinet_dof_props['stiffness'][0] = 30.0
        cabinet_dof_props['friction'][0] = 2.0
        cabinet_dof_props['effort'][0] = 4.0
        cabinet_dof_props['driveMode'][0] = gymapi.DOF_MODE_POS
        self.gym.set_actor_dof_properties(env_ptr, obj_actor, cabinet_dof_props)
        self.cabinet_actor_list.append(obj_actor)

        hazard_actor = self.gym.create_actor(
            env_ptr,
            self.hazard_asset,
            self.hazard_pose,
            f'hazard-{env_id}',
            env_id,  # collision group
            3,  # filter
            0,
        )
        # set hazard area as red
        self.gym.set_rigid_body_color(
            env_ptr,
            hazard_actor,
            self.gym.find_asset_rigid_body_index(self.hazard_asset, 'box'),
            gymapi.MESH_VISUAL_AND_COLLISION,
            gymapi.Vec3(1.0, 0.0, 0.0),
        )

    def _place_agents(self, env_num, spacing):
        print('Simulator: creating agents')

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((env_num, 3), device=self.device)
        self.space_range = torch.zeros((env_num, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing / 2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing / 2
        num_per_row = int(np.sqrt(env_num))

        with tqdm(total=env_num) as pbar:
            pbar.set_description('Enumerating envs:')
            for env_id in range(env_num):
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                self.env_ptr_list.append(env_ptr)
                self._load_franka(env_ptr, env_id)
                self._load_obj(env_ptr, env_id)
                pbar.update(1)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _get_reward_cost_done(self):
        door_pos = self.cabinet_door_rigid_body_tensor[:, :3]
        door_rot = self.cabinet_door_rigid_body_tensor[:, 3:7]
        hand_rot = self.hand_rigid_body_tensor[..., 3:7]
        hand_down_dir = quat_axis(hand_rot, 2)
        hand_grip_dir = quat_axis(hand_rot, 1)
        hand_sep_dir = quat_axis(hand_rot, 0)
        handle_pos = quat_apply(door_rot, self.cabinet_handle_pos_tensor) + door_pos
        handle_x = quat_axis(door_rot, 0) * self.cabinet_open_dir_tensor.view(-1, 1)
        handle_z = quat_axis(door_rot, 1)

        cabinet_door_relative_o = door_pos + quat_apply(door_rot, self.cabinet_door_edge_min)
        cabinet_door_relative_x = -handle_x
        cabinet_door_relative_y = -quat_axis(door_rot, 2)
        cabinet_door_relative_z = quat_axis(door_rot, 1)

        franka_rfinger_pos = (
            self.rigid_body_tensor[:, self.hand_lfinger_rigid_body_index][:, 0:3]
            + hand_down_dir * 0.075
        )
        franka_lfinger_pos = (
            self.rigid_body_tensor[:, self.hand_rfinger_rigid_body_index][:, 0:3]
            + hand_down_dir * 0.075
        )

        freight_pos = self.rigid_body_tensor[:, self.freight_rigid_body_index][:, 0:3]

        # close door or drawer
        door_reward = self.cabinet_dof_coef * self.cabinet_dof_tensor[:, 0]
        action_penalty = torch.sum(
            (self.pos_act[:, :7] - self.franka_dof_tensor[:, :7, 0]) ** 2, dim=-1
        )

        d = torch.norm(self.hand_tip_pos - handle_pos, p=2, dim=-1)

        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.1, dist_reward * 2, dist_reward)
        dist_reward *= self.cabinet_have_handle_tensor

        dot1 = (hand_grip_dir * handle_z).sum(dim=-1)
        dot2 = (-hand_sep_dir * handle_x).sum(dim=-1)
        # reward for matching the orientation of the hand to the drawer (fingers wrapped)
        rot_reward = 0.5 * (torch.sign(dot1) * dot1**2 + torch.sign(dot2) * dot2**2)

        diff_from_success = torch.abs(
            self.cabinet_dof_tensor_spec[:, :, 0]
            - self.success_dof_states.view(self.cabinet_num, -1)
        ).view(-1)
        success = diff_from_success < 0.01
        success_bonus = success

        open_reward = self.cabinet_dof_tensor[:, 0] * 10

        self.rew_buf = 1.0 * dist_reward + 0.5 * rot_reward - 1 * open_reward

        self.cost_x_range = torch.tensor([-0.2, 0.3])
        self.cost_y_range = torch.tensor([-0.5, 0.5])

        freight_x = freight_pos[:, 0]
        freight_y = freight_pos[:, 1]

        within_x = (self.cost_x_range[0] <= freight_x) & (freight_x <= self.cost_x_range[1])
        within_y = (self.cost_y_range[0] <= freight_y) & (freight_y <= self.cost_y_range[1])

        self.cost_buf = (within_x & within_y).type(torch.float32)

        # set the type

        time_out = self.progress_buf >= self.max_episode_length
        self.reset_buf = self.reset_buf | time_out
        self.success_buf = self.success_buf | success
        self.success = self.success_buf & time_out

        old_coef = 1.0 - time_out * 0.1
        new_coef = time_out * 0.1

        self.success_rate = self.success_rate * old_coef + success * new_coef

        return self.rew_buf, self.cost_buf, self.reset_buf

    def _get_base_observation(self, suggested_gt=None):
        hand_rot = self.hand_rigid_body_tensor[..., 3:7]
        hand_down_dir = quat_axis(hand_rot, 2)
        self.hand_tip_pos = (
            self.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * self.gripper_length
        )  # calculating middle of two fingers
        self.hand_rot = hand_rot

        dim = 57

        state = torch.zeros((self.num_envs, dim), device=self.device)

        joints = self.franka_num_dofs
        # joint dof value
        state[:, :joints].copy_(
            (
                2
                * (
                    self.franka_dof_tensor[:, :joints, 0]
                    - self.franka_dof_lower_limits_tensor[:joints]
                )
                / (
                    self.franka_dof_upper_limits_tensor[:joints]
                    - self.franka_dof_lower_limits_tensor[:joints]
                )
            )
            - 1
        )
        # joint dof velocity
        state[:, joints : joints * 2].copy_(self.franka_dof_tensor[:, :joints, 1])
        # cabinet dof
        state[:, joints * 2 : joints * 2 + 2].copy_(self.cabinet_dof_tensor)
        # hand
        state[:, joints * 2 + 2 : joints * 2 + 15].copy_(
            relative_pose(self.franka_root_tensor, self.hand_rigid_body_tensor).view(
                self.env_num, -1
            )
        )
        # actions
        state[:, joints * 2 + 15 : joints * 3 + 15].copy_(self.actions[:, :joints])

        state[:, joints * 3 + 15 : joints * 3 + 15 + 3].copy_(
            self.franka_root_tensor[:, 0:3] - self.cabinet_handle_pos_tensor
        )
        state[:, joints * 3 + 15 + 3 : joints * 3 + 15 + 3 + 3].copy_(
            self.cabinet_handle_pos_tensor - self.hand_tip_pos
        )

        return state

    def _refresh_observation(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.obs_buf = self._get_base_observation()

    def _perform_actions(self, actions):
        actions = actions.to(self.device)
        self.actions = actions

        joints = self.franka_num_dofs - 2
        self.pos_act[:, :-3] = (
            self.pos_act[:, :-3] + actions[:, 0:joints] * self.dt * self.action_speed_scale
        )
        self.pos_act[:, :joints] = tensor_clamp(
            self.pos_act[:, :joints],
            self.franka_dof_lower_limits_tensor[:joints],
            self.franka_dof_upper_limits_tensor[:joints],
        )

        self.eff_act[:, -3] = actions[:, -2] * self.franka_dof_max_torque_tensor[-2]  # gripper1
        self.eff_act[:, -2] = actions[:, -1] * self.franka_dof_max_torque_tensor[-1]  # gripper2
        self.pos_act[:, self.franka_num_dofs] = self.cabinet_dof_target  # door reverse force
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.pos_act.view(-1))
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self.eff_act.view(-1))
        )

    def _draw_line(self, src, dst):
        line_vec = np.stack([src.cpu().numpy(), dst.cpu().numpy()]).flatten().astype(np.float32)
        color = np.array([1, 0, 0], dtype=np.float32)
        self.gym.clear_lines(self.viewer)
        self.gym.add_lines(self.viewer, self.env_ptr_list[0], self.env_num, line_vec, color)

    # @TimeCounter
    def step(self, actions):
        self._perform_actions(actions)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        self.env_frame_idx += 1
        self.render()

        self.progress_buf += 1

        self._refresh_observation()

        reward, cost, done = self._get_reward_cost_done()

        done = self.reset_buf.clone()
        success = self.success.clone()
        self._partial_reset(self.reset_buf)

        if self.average_reward == None:
            self.average_reward = self.rew_buf.mean()
        else:
            self.average_reward = self.rew_buf.mean() * 0.01 + self.average_reward * 0.99
        self.extras['successes'] = success
        self.extras['success_rate'] = self.success_rate
        return self.obs_buf, self.rew_buf, self.cost_buf, done, None

    def _partial_reset(self, to_reset='all'):
        """
        reset those need to be reseted
        """

        if to_reset == 'all':
            to_reset = np.ones((self.env_num,))
        reseted = False
        for env_id, reset in enumerate(to_reset):
            # is reset:
            if reset.item():
                # need randomization
                reset_dof_states = self.initial_dof_states[env_id].clone()
                reset_root_states = self.initial_root_states[env_id].clone()
                franka_reset_pos_tensor = reset_root_states[0, :3]
                franka_reset_rot_tensor = reset_root_states[0, 3:7]
                franka_reset_dof_pos_tensor = reset_dof_states[: self.franka_num_dofs, 0]
                franka_reset_dof_vel_tensor = reset_dof_states[: self.franka_num_dofs, 1]
                cabinet_reset_pos_tensor = reset_root_states[1, :3]
                cabinet_reset_rot_tensor = reset_root_states[1, 3:7]
                cabinet_reset_dof_pos_tensor = reset_dof_states[self.franka_num_dofs :, 0]
                cabinet_reset_dof_vel_tensor = reset_dof_states[self.franka_num_dofs :, 1]

                cabinet_type = env_id // self.env_per_cabinet

                self.intervaledRandom_(franka_reset_pos_tensor, self.franka_reset_position_noise)
                self.intervaledRandom_(franka_reset_rot_tensor, self.franka_reset_rotation_noise)
                self.intervaledRandom_(
                    franka_reset_dof_pos_tensor,
                    self.franka_reset_dof_pos_interval,
                    self.franka_dof_lower_limits_tensor,
                    self.franka_dof_upper_limits_tensor,
                )
                self.intervaledRandom_(
                    franka_reset_dof_vel_tensor, self.franka_reset_dof_vel_interval
                )
                self.intervaledRandom_(cabinet_reset_pos_tensor, self.cabinet_reset_position_noise)
                self.intervaledRandom_(cabinet_reset_rot_tensor, self.cabinet_reset_rotation_noise)
                self.intervaledRandom_(
                    cabinet_reset_dof_pos_tensor,
                    self.cabinet_reset_dof_pos_interval,
                    self.cabinet_dof_lower_limits_tensor[cabinet_type],
                    self.cabinet_dof_upper_limits_tensor[cabinet_type],
                )
                self.intervaledRandom_(
                    cabinet_reset_dof_vel_tensor, self.cabinet_reset_dof_vel_interval
                )

                self.dof_state_tensor[env_id].copy_(reset_dof_states)
                self.root_tensor[env_id].copy_(reset_root_states)
                reseted = True
                self.progress_buf[env_id] = 0
                self.reset_buf[env_id] = 0
                self.success_buf[env_id] = 0

                # Reset the frame index for this environment
                self.env_frame_idx[env_id] = 0

                # Increment the episode index for this environment
                self.env_episode_idx[env_id] += 1

        if reseted:
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))

    def reset(self, to_reset='all'):
        self._partial_reset(to_reset)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless:
            self.render()
        if self.cfg['env']['enableCameraSensors'] == True:
            self.gym.step_graphics(self.sim)

        self._refresh_observation()
        success = self.success.clone()
        reward, cost, done = self._get_reward_cost_done()

        self.extras['successes'] = success
        self.extras['success_rate'] = self.success_rate
        return self.obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, None

    def save(self, path, iteration):
        pass

    def update(self, it=0):
        pass

    def train(self):  # changing mode to train
        self.train_mode = True

    def eval(self):  # changing mode to eval
        self.train_mode = False

    def _get_cabinet_door_mask(self):
        door_pos = self.cabinet_door_rigid_body_tensor[:, :3]
        door_rot = self.cabinet_door_rigid_body_tensor[:, 3:7]

        handle_x = quat_axis(door_rot, 0) * self.cabinet_open_dir_tensor.view(-1, 1)

        cabinet_door_edge_min = door_pos + quat_apply(door_rot, self.cabinet_door_edge_min)

        cabinet_door_relative_o = cabinet_door_edge_min
        cabinet_door_relative_x = handle_x
        cabinet_door_relative_y = quat_axis(door_rot, 2)
        cabinet_door_relative_z = quat_axis(door_rot, 1)

        cabinet_door_x = self.cabinet_door_max_tensor[:, 0] - self.cabinet_door_min_tensor[:, 0]
        cabinet_door_y = self.cabinet_door_max_tensor[:, 2] - self.cabinet_door_min_tensor[:, 2]
        cabinet_door_z = self.cabinet_door_max_tensor[:, 1] - self.cabinet_door_min_tensor[:, 1]

        return (
            cabinet_door_relative_o,  # origin coord
            cabinet_door_relative_x,  # normalized x axis
            cabinet_door_relative_y,
            cabinet_door_relative_z,
            cabinet_door_x,  # length of x axis
            cabinet_door_y,
            cabinet_door_z,
        )

    def _detailed_view(self, tensor):
        shape = tensor.shape
        return tensor.view(self.cabinet_num, self.env_per_cabinet, *shape[1:])

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None):
        tensor += torch.rand(tensor.shape, device=self.device) * dist * 2 - dist
        if lower is not None and upper is not None:
            torch.clamp_(tensor, min=lower, max=upper)
            
    def render(self):
        # Step the graphics for rendering
        self.gym.step_graphics(self.sim)

        # Render all camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        # Loop over each environment and capture images
        for env_id, env in enumerate(self.env_ptr_list):
            camera_handle = self.cameras[env_id]

            # Get the RGBA image tensor from the camera
            image_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, env, camera_handle, gymapi.IMAGE_COLOR
            )
            torch_image_tensor = gymtorch.wrap_tensor(image_tensor)

            # Copy the tensor to CPU and convert to NumPy array
            img = torch_image_tensor.cpu().numpy()

            # Convert image from RGBA to RGB and flip vertically
            img_rgb = img[..., :3]
            img_rgb = np.flipud(img_rgb).astype(np.uint8)

            # Build the output path
            env_dir = self.output_dirs[env_id]
            episode_idx = self.env_episode_idx[env_id].item()
            frame_idx = self.env_frame_idx[env_id].item()
            image_path = os.path.join(env_dir, f'episode_{episode_idx}_frame_{frame_idx}.png')

            # Save the image as PNG
            Image.fromarray(img_rgb).save(image_path)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def control_ik(j_eef, device, dpose, num_envs):
    # Set controller parameters
    # IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u


def relative_pose(src, dst):
    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    ip = dst.view(-1, shape[-1])[:, 3:]
    ret = torch.cat((p, ip), dim=1)
    return ret.view(*shape)



# VecEnv Wrapper for FreightFranka
class FreightFrankaMultiVecTask:
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_states = task.num_states
        self.num_actions = task.num_actions
        self.num_freight_obs = task.num_freight_obs
        self.num_franka_obs = task.num_franka_obs

        self.num_freight_observations = task.num_obs - self.num_franka_obs
        self.num_franka_observations = task.num_obs - self.num_freight_obs
        self.nums_share_observations = task.num_obs
        self.num_agents = 2

        self.clip_actions_low = task.franka_dof_lower_limits_tensor
        self.clip_actions_high = task.franka_dof_upper_limits_tensor
        self.rl_device = rl_device
        print('RL device: ', rl_device)

        # COMPATIBILITY
        # self.observation_space = [Box(low=np.array([-10]*self.n_agents), high=np.array([10]*self.n_agents)) for _ in range(self.n_agents)]
        self.obs_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_freight_observations,)),
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_franka_observations,)),
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.nums_share_observations,))
            for _ in range(self.num_agents)
        ]

        self.act_space = tuple(
            [
                spaces.Box(
                    low=np.ones((3,)) * self.clip_actions_low[:3].cpu().numpy(),
                    high=np.ones((3,)) * self.clip_actions_high[:3].cpu().numpy(),
                ),
                spaces.Box(
                    low=np.ones((9,)) * self.clip_actions_low[3:12].cpu().numpy(),
                    high=np.ones((9,)) * self.clip_actions_high[3:12].cpu().numpy(),
                ),
            ]
        )

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        env_info = {
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
            'n_actions': self.get_total_actions(),
            'n_agents': self.num_agents,
        }
        return env_info

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


# Python CPU/GPU Class
class FreightFrankaMultiVecTaskPython(FreightFrankaMultiVecTask):
    def get_state(self):
        return self.task.states_buf.to(self.rl_device)

    def step(self, actions):
        an_agent_actions = actions[0]
        for i in range(1, len(actions)):
            an_agent_actions = torch.hstack((an_agent_actions, actions[i]))

        actions = an_agent_actions

        actions_tensor = torch.clamp(actions, self.clip_actions_low, self.clip_actions_high)

        obs_buf, rew_buf, cost_buf, reset_buf, _ = self.task.step(actions_tensor)

        sub_agent_obs = []
        # self.process_sub_agent_obs(self.agent_dof_index, self.agent_finger_index, obs_buf)
        num_freight_obs = self.num_freight_obs // 2
        num_franka_obs = self.num_franka_obs // 2
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, :num_freight_obs],
                    obs_buf[:, 12 : 12 + num_freight_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, num_freight_obs : num_freight_obs + num_franka_obs],
                    obs_buf[:, 12 + num_freight_obs : 12 + num_freight_obs + num_franka_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        state_buf = obs_buf
        rewards = rew_buf.unsqueeze(-1).to(self.rl_device)
        costs = cost_buf.unsqueeze(-1).to(self.rl_device)
        dones = reset_buf.to(self.rl_device)

        agent_state = []
        sub_agent_reward = []
        sub_agent_cost = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(2):
            agent_state.append(state_buf)
            sub_agent_reward.append(rewards)
            sub_agent_cost.append(costs)
            sub_agent_done.append(dones)
            sub_agent_info.append(torch.Tensor(0))

        # obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        obs_all = sub_agent_obs
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        reward_all = torch.transpose(torch.stack(sub_agent_reward), 1, 0)
        costs_all = torch.transpose(torch.stack(sub_agent_cost), 1, 0)
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        info_all = torch.stack(sub_agent_info)

        return obs_all, state_all, reward_all, costs_all, done_all, info_all, None

    def reset(self):
        actions = 0.01 * (
            1
            - 2
            * torch.rand(
                [self.task.num_envs, self.task.num_actions],
                dtype=torch.float32,
                device=self.rl_device,
            )
        )

        # step the simulator
        self.task.step(actions)

        sub_agent_obs = []
        obs_buf = self.task.obs_buf
        # self.process_sub_agent_obs(self.agent_dof_index, self.agent_finger_index, obs_buf)
        num_freight_obs = self.num_freight_obs // 2
        num_franka_obs = self.num_franka_obs // 2
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, :num_freight_obs],
                    obs_buf[:, 12 : 12 + num_freight_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        sub_agent_obs.append(
            torch.cat(
                [
                    obs_buf[:, num_freight_obs : num_freight_obs + num_franka_obs],
                    obs_buf[:, 12 + num_freight_obs : 12 + num_freight_obs + num_franka_obs],
                    obs_buf[:, 24:],
                ],
                dim=1,
            )
        )
        state_buf = self.task.obs_buf

        agent_state = []
        for i in range(2):
            agent_state.append(state_buf)

        # obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        obs_all = sub_agent_obs
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs_all, state_all, None


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim

    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def make_ma_isaac_env(args, cfg, cfg_train, sim_params, agent_index):
    """
    Creates and returns a multi-agent environment for the Isaac Gym task.

    Args:
        args: Command-line arguments.
        cfg: Configuration for the environment.
        cfg_train: Training configuration.
        sim_params: Parameters for the simulation.
        agent_index: Index of the agent within the multi-agent environment.

    Returns:
        env: A multi-agent environment for the Isaac Gym task.
    """
    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]   
    cfg_task["seed"] = cfg["seed"]
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        agent_index=agent_index,
        is_multi_agent=True)
    task_name = task.__class__.__name__
    if "ShadowHand" in task_name:
        raise NotImplementedError
    elif "FreightFranka" in task_name:
        env = FreightFrankaMultiVecTaskPython(task, rl_device)
    else:
        raise NotImplementedError

    return env

def eval_multi_agent(eval_dir, eval_episodes):

    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    env_name = config['env_name']
    if env_name in isaac_gym_map:
        args_cfg_path = eval_dir + '/env_cfg/args.pkl'
        cfg_env_path = eval_dir + '/env_cfg/cfg_env.json'
        cfg_train_path = eval_dir + '/env_cfg/cfg_train.json'
        agent_index = [[[0, 1, 2, 3, 4, 5]],
                       [[0, 1, 2, 3, 4, 5]]]
        
        train_args = pkl.load(open(args_cfg_path, 'rb'))
        cfg_env = json.load(open(cfg_env_path, 'r'))
        cfg_train = json.load(open(cfg_train_path, 'r'))
        
        cfg_env['env']['numEnvs'] = 10
        cfg_train['n_eval_rollout_threads'] = 10
        cfg_train['n_rollout_threads'] = 10
        
        config['n_eval_rollout_threads'] = 10
        config['n_rollout_threads'] = 10
        # cfg_env['env']['headless'] = True
        # cfg_env['env']['device_id'] = 1
        # cfg_train['n_rollout_threads'] = 1
        # cfg_train['n_eval_rollout_threads'] = 1
        
        sim_params = parse_sim_params(train_args, cfg_env, cfg_train)
        env = make_ma_isaac_env(train_args, cfg_env, cfg_train, sim_params, agent_index)
        eval_env = env
    else:
        raise NotImplementedError


    model_dir = eval_dir + f"/models_seed{config['seed']}"
    algo = config['algorithm_name']
    if algo == 'macpo':
        from safepo.multi_agent.macpo import Runner
    elif algo == 'mappo':
        from safepo.multi_agent.mappo import Runner
    elif algo == 'mappolag':
        from safepo.multi_agent.mappolag import Runner
    elif algo == 'happo':
        from safepo.multi_agent.happo import Runner
    else:
        raise NotImplementedError
    torch.set_num_threads(4)
    
    runner = Runner(
        vec_env=eval_env,
        vec_eval_env=eval_env,
        config=config,
        model_dir=model_dir,
    )
    
    res = runner.eval(eval_episodes)

    env.task.close()
    
    return res

def benchmark_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, default='', help="the directory of the evaluation")
    parser.add_argument("--eval-episodes", type=int, default=1, help="the number of episodes to evaluate")
    parser.add_argument("--save-dir", type=str, default=None, help="the directory to save the evaluation result")

    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir
    eval_episodes = args.eval_episodes
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = benchmark_dir.replace('runs', 'results')
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
    envs = os.listdir(benchmark_dir)
    for env in envs:
        env_path = os.path.join(benchmark_dir, env)
        algos = os.listdir(env_path)
        for algo in algos:
            print(f"Start evaluating {algo} in {env}")
            algo_path = os.path.join(env_path, algo)
            seeds = os.listdir(algo_path)
            rewards, costs = [], []
            for seed in seeds:
                print("so far so good")
                seed_path = os.path.join(algo_path, seed)
                reward, cost = eval_multi_agent(seed_path, eval_episodes)
                rewards.append(reward)
                costs.append(cost)
                
            output_file = open(f"{save_dir}/eval_result.txt", 'a')
            
            # two wise after point
            reward_mean = round(np.mean(rewards), 2)
            reward_std = round(np.std(rewards), 2)
            cost_mean = round(np.mean(costs), 2)
            cost_std = round(np.std(costs), 2)
            print(f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std}, the reuslt is saved in {save_dir}/eval_result.txt")
            # output_file.write(f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}±{reward_std}, cost: {cost_mean}±{cost_std} \n")
            output_file.write(f"After {eval_episodes} episodes evaluation, the {algo} in {env} evaluation reward: {reward_mean}+/-{reward_std}, cost: {cost_mean}+/-{cost_std} \n")


if __name__ == '__main__':
    benchmark_eval()
