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

import argparse
import os
import json
from safepo.common.env import make_sa_mujoco_env
from safepo.common.model import ActorVCritic
import numpy as np
import joblib
import torch
import mujoco

from tqdm import tqdm
import imageio
import pickle
import multiprocessing as mp

from dataclasses import dataclass
# from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from safety_gymnasium.builder import Builder



from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.utils.keyboard_viewer import KeyboardViewer
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from safety_gymnasium.bases.underlying import (
    RenderConf,
    SimulationConf,
    VisionEnvConf,
    FloorConf,
    PlacementsConf,
    WorldInfo,
    RandomGenerator,
)

def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype)


class Renderer:  # pylint: disable=too-many-instance-attributes
    # def __init__(self, config: dict | None = None) -> None:
    def __init__(self, env) -> None:
        """Initialize the engine.

        Args:
            config (dict): Configuration dictionary, used to pre-config some attributes
                according to tasks via :meth:`safety_gymnasium.register`.
        """

        # self.sim_conf = SimulationConf()
        # self.placements_conf = PlacementsConf()
        # self.render_conf = RenderConf()
        # self.vision_env_conf = VisionEnvConf()
        # self.floor_conf = FloorConf()

        # self.random_generator = RandomGenerator()

        self.viewer = None
        self._viewers = {}
        
        self.model = env.model
        self.data = env.data

        # # Obstacles which are added in environments.
        # self._geoms = {}
        # self._free_geoms = {}
        # self._mocaps = {}

        # self._parse(config)
        
    # pylint: disable-next=too-many-arguments,too-many-branches,too-many-statements
    def render(
        self,
        width: int,
        height: int,
        mode: str,
        camera_id: int = None,
        camera_name: str = None,
        cost: float = None,
    ) -> None:
        """Render the environment to somewhere.

        Note:
            The camera_name parameter can be chosen from:
                - **human**: the camera used for freely moving around and can get input
                from keyboard real time.
                - **vision**: the camera used for vision observation, which is fixed in front of the
                agent's head.
                - **track**: The camera used for tracking the agent.
                - **fixednear**: the camera used for top-down observation.
                - **fixedfar**: the camera used for top-down observation, but is further than **fixednear**.
        """
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height

        if mode in {
            'rgb_array',
            'depth_array',
        }:
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    'Both `camera_id` and `camera_name` cannot be specified at the same time.',
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'vision'

            if camera_id is None:
                # pylint: disable-next=no-member
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,  # pylint: disable=no-member
                    camera_name,
                )

        self._get_viewer(mode)

        # Turn all the geom groups on
        self.viewer.vopt.geomgroup[:] = 1

        # # Lidar and Compass markers
        # if self.render_conf.lidar_markers:
        #     offset = (
        #         self.render_conf.lidar_offset_init
        #     )  # Height offset for successive lidar indicators
        #     for obstacle in self._obstacles:
        #         if obstacle.is_lidar_observed:
        #             self._render_lidar(obstacle.pos, obstacle.color, offset, obstacle.group)
        #         if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
        #             self._render_compass(
        #                 getattr(self, obstacle.name + '_pos'),
        #                 obstacle.color,
        #                 offset,
        #             )
        #         offset += self.render_conf.lidar_offset_delta

        # Add indicator for nonzero cost
        # if cost.get('cost_sum', 0) > 0:
        #     self._render_sphere(self.agent.pos, 0.25, COLOR['red'], alpha=0.5)

        # Draw vision pixels
        if mode in {'rgb_array', 'depth_array'}:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).render(render_mode=mode, camera_id=camera_id)
            self.viewer._markers[:] = []  # pylint: disable=protected-access
            self.viewer._overlays.clear()  # pylint: disable=protected-access
            return data
        
        if mode == 'human':
            self._get_viewer(mode).render()
            return None
        
        raise NotImplementedError(f'Render mode {mode} is not implemented.')

    def _get_viewer(
        self,
        mode: str,
    ):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = KeyboardViewer(
                    self.model,
                    self.data,
                    self.agent.keyboard_control_callback,
                )
            elif mode in {'rgb_array', 'depth_array'}:
                self.viewer = OffScreenViewer(self.model, self.data)
            else:
                raise AttributeError(f'Unexpected mode: {mode}')

            # self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer
    
    # pylint: disable-next=too-many-arguments
    def _render_sphere(
        self,
        pos: np.ndarray,
        size: float,
        color: np.ndarray,
        label: str = '',
        alpha: float = 0.1,
    ) -> None:
        """Render a radial area in the environment."""
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=size * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
            rgba=np.array(color) * alpha,
            label=label if self.render_conf.labels else '',
        )
        

def collect_video(obs, env, model, renderer, height, width, camera, mode):
    images = []
    actions = []
    rewards = []
    costs = []
    observations = [obs]
    episode_return = 0
    done = False
    image = renderer.render(camera_name=camera, height=height, width=width, mode=mode)
    images += [image]
    
    done = False
    while True:
        with torch.no_grad():
            action, _, _, _ = model.step(
                to_tensor(obs), deterministic=True
            )
        
        try:
            obs, reward, cost, terminated, truncated, info = env.step(
                    action.detach().squeeze().cpu().numpy()
                )
            done = terminated[0] or truncated[0]
            episode_return += reward
            observations += [obs]
            actions += [action]
            rewards += [reward[0]]
            costs += [cost[0]]
            #print(i, done, dd)
        except Exception as e:
            print(e)
            break
        if done:
            break
        image = renderer.render(camera_name=camera, height=height, width=width, mode=mode)
        images += [image]
        
    return observations, images, actions, rewards, costs, episode_return


def sample_n_frames(frames, n):
    new_vid_ind = [int(i*len(frames)/(n-1)) for i in range(n-1)] + [len(frames)-1]
    return np.array([frames[i] for i in new_vid_ind])


def save_frame(path, frame):
    imageio.imwrite(path, frame)
    
if __name__ == '__main__':
    
    collection_config = {
        # "camera_names": ['vision', 'track', 'fixednear', 'fixedfar'], ### possible values: "corner3, corner, corner2, topview, behindGripper", None(for random)
        "camera_names": ['vision', 'track', 'fixednear'], ### possible values: "corner3, corner, corner2, topview, behindGripper", None(for random)
        "mode": "rgb_array", ### possible values: "rgb_array", "depth_array", "human"
        "discard_ratio": 0.0, ### discard the last {ratio} of the collected videos (preventing failed episodes)
        "height": 512,
        "width": 512,
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default='', help="the directory of the evaluation")
    parser.add_argument("--episodes_per_camera", type=int, default=25, help="the number of episodes to evaluate")

    args = parser.parse_args()

    env_path = args.exp_dir
    episodes_per_cam = args.episodes_per_camera
    save_dir = env_path.replace('runs', 'collected_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # torch.set_num_threads(4)
    
    config_path = env_path + '/config.json'
    config = json.load(open(config_path, 'r'))

    env_id = config['task'] if 'task' in config.keys() else config['env_name']
    env_norms = os.listdir(env_path)
    env_norms = [env_norm for env_norm in env_norms if env_norm.endswith('.pkl')]
    final_norm_name = sorted(env_norms)[-1]

    model_dir = env_path + '/torch_save'
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith('.pt')]
    final_model_name = sorted(models)[-1]

    model_path = model_dir + '/' + final_model_name
    norm_path = env_path + '/' + final_norm_name

    for camera in collection_config["camera_names"]:
        demos = []
        action_seqs = []
        raw_lengths = []
        rewards = []
        costs = []
        for seed in tqdm(range(42, 42+episodes_per_cam)):
            
            eval_env, obs_space, act_space = make_sa_mujoco_env(
                num_envs=1, env_id=env_id, seed=seed
            )
            
            renderer = Renderer(eval_env)

            model = ActorVCritic(
                    obs_dim=obs_space.shape[0],
                    act_dim=act_space.shape[0],
                    hidden_sizes=config['hidden_sizes'],
                )
            model.actor.load_state_dict(torch.load(model_path))

            if os.path.exists(norm_path):
                norm = joblib.load(open(norm_path, 'rb'))['Normalizer']
                eval_env.obs_rms = norm
            
            obs, _ = eval_env.reset()
            observations, images, action_seq, reward, cost, _ = collect_video(
                obs, eval_env, model, renderer, 
                height=collection_config["height"], width=collection_config["width"],
                camera=camera, mode=collection_config["mode"]
            )
            
            # assert len(images) == len(action_seq) + 1 or len(images) == 502
            raw_lengths += [len(images)]
            demos += [images]
            action_seqs += [action_seq]
            rewards += [reward]
            costs += [cost]
            
        top_k_ind = np.argsort(raw_lengths)[:episodes_per_cam]
        demos = [demos[i] for i in top_k_ind]
        raw_lengths = [raw_lengths[i] for i in top_k_ind]
        action_seqs = [action_seqs[i] for i in top_k_ind]
        rewards = [rewards[i] for i in top_k_ind]
        costs = [costs[i] for i in top_k_ind]
        observations = [observations[i] for i in top_k_ind]
        print(f"vid length bounds: {raw_lengths[0]} ~ {raw_lengths[-1]}")
        
        ### save the collected demos
        print(f"Saving the collected demos to {save_dir}/{camera}")
        for i, demo in enumerate(demos):
            demo_dir = os.path.join(save_dir, f"{camera}/{i:03d}")
            os.makedirs(demo_dir, exist_ok=True)
            with mp.Pool(10) as p:
                p.starmap(save_frame, [(os.path.join(demo_dir, f"{j:02d}.png"), frame) for j, frame in enumerate(demo)])
            with open(f"{demo_dir}/observations.pkl", "wb") as f:
                pickle.dump(observations[i], f)
            with open(f"{demo_dir}/action.pkl", "wb") as f:
                pickle.dump(action_seqs[i], f)
            with open(f"{demo_dir}/rewards.pkl", "wb") as f:
                    pickle.dump(rewards[i], f)
            with open(f"{demo_dir}/costs.pkl", "wb") as f:
                    pickle.dump(costs[i], f)
            
            