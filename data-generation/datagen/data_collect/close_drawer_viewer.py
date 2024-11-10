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

try:
    from isaacgym import gymapi, gymutil
except ImportError:
    raise Exception("Please install isaacgym to run Isaac Gym tasks!")

import argparse
import os
import json
from collections import deque
from safepo.common.env import make_sa_mujoco_env, make_ma_mujoco_env, make_ma_multi_goal_env, make_ma_isaac_env
from safepo.common.model import ActorVCritic
from safepo.utils.config import multi_agent_velocity_map, multi_agent_goal_tasks, isaac_gym_map
from safepo.utils.config import multi_agent_args, set_seed
import numpy as np
import torch
import pickle as pkl

from argparse import Namespace

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

def eval_multi_agent(eval_dir, eval_episodes):

    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    env_name = config['env_name']
    if env_name in multi_agent_velocity_map.keys():
        env_info = multi_agent_velocity_map[env_name]
        agent_conf = env_info['agent_conf']
        scenario = env_info['scenario']
        eval_env = make_ma_mujoco_env(
            scenario=scenario,
            agent_conf=agent_conf,
            seed=np.random.randint(0, 1000),
            cfg_train=config,
        )
    elif env_name in isaac_gym_map:
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
        cfg_train['log_dir'] = eval_dir
        
        config['n_eval_rollout_threads'] = 10
        config['n_rollout_threads'] = 10
        config['log_dir'] = eval_dir
        
        sim_params = parse_sim_params(train_args, cfg_env, cfg_train)
        env = make_ma_isaac_env(train_args, cfg_env, cfg_train, sim_params, agent_index)
        eval_env = env
    else:
        eval_env = make_ma_multi_goal_env(
            task=env_name,
            seed=np.random.randint(0, 1000),
            cfg_train=config,
        )

    model_dir = eval_dir + f"/models_seed{config['seed']}"
    algo = config['algorithm_name']
    if algo == 'macpo':
        from datagen.train_scripts.macpo import Runner
    elif algo == 'mappo':
        from datagen.train_scripts.mappo import Runner
    elif algo == 'mappolag':
        from datagen.train_scripts.mappolag import Runner
    elif algo == 'happo':
        from datagen.train_scripts.happo import Runner
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
