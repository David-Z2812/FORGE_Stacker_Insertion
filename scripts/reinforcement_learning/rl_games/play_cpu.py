# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import random
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    # agent.restore(resume_path)
    
    # --- load checkpoint and force everything to CPU (safe for rl_games checkpoints) ---
    checkpoint = torch.load(resume_path, map_location=torch.device("cpu"), weights_only=False)
    agent.model.load_state_dict(checkpoint["model"])
    print("[INFO]: Loaded checkpoint to CPU directly.")

    # Move top-level model to CPU (module.to handles parameters/buffers for the module)
    agent.model.to(torch.device("cpu"))
    print("[INFO]: Moved model to CPU (top-level).")
    
    # --- Force rl-games scaling tensors to CPU ---
    if hasattr(agent, "actions_low"):
        agent.actions_low = agent.actions_low.to("cpu")
    if hasattr(agent, "actions_high"):
        agent.actions_high = agent.actions_high.to("cpu")
    if hasattr(agent, "value_mean_std"):
        for attr in ["mean", "var"]:
            if hasattr(agent.value_mean_std, attr):
                setattr(agent.value_mean_std, attr, getattr(agent.value_mean_std, attr).to("cpu"))

    print("[INFO]: Moved all scaling tensors (actions_low/high, value_mean_std) to CPU.")

    
    # ALSO ensure all parameters/buffers are on cpu (defensive):
    for name, param in agent.model.named_parameters(recurse=True):
        if param is not None:
            param.data = param.data.cpu()
            if param.grad is not None:
                param.grad.data = param.grad.data.cpu()
    for name, buf in agent.model.named_buffers(recurse=True):
        if buf is not None:
            buf.data = buf.data.cpu()

    # Move nested rl-games network pieces to CPU (a2c_network and its rnn)
    if hasattr(agent.model, "a2c_network"):
        agent.model.a2c_network.to(torch.device("cpu"))
        print("[INFO]: Moved agent.model.a2c_network to CPU.")
        if hasattr(agent.model.a2c_network, "rnn"):
            agent.model.a2c_network.rnn.to(torch.device("cpu"))
            print("[INFO]: Moved internal RNN (a2c_network.rnn) to CPU.")
      
    # --- Force rl-games internal device variables to CPU ---
    if hasattr(agent.model, "a2c_network"):
        net = agent.model.a2c_network
        if hasattr(net, "device_name"):
            net.device_name = "cpu"
            print(f"[INFO]: Forced a2c_network.device_name = {net.device_name}")
        if hasattr(net, "rnn") and hasattr(net.rnn, "device"):
            net.rnn.device = torch.device("cpu")
            print("[INFO]: Forced RNN internal device to CPU.")

    # Some rl-games versions cache device in the parent model too
    if hasattr(agent.model, "device"):
        agent.model.device = torch.device("cpu")
        print("[INFO]: Forced agent.model.device to CPU.")


    # Force all hidden states in agent (RNN states) to CPU
    if hasattr(agent, "states") and agent.states is not None:
        # agent.states can be list/tuple/dict/tensor depending on rl-games version
        if isinstance(agent.states, (list, tuple)):
            agent.states = [s.to("cpu") if torch.is_tensor(s) else s for s in agent.states]
        elif isinstance(agent.states, dict):
            agent.states = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in agent.states.items()}
        elif torch.is_tensor(agent.states):
            agent.states = agent.states.to("cpu")
        print("[INFO]: Force-moved agent.states (hidden states) to CPU.")

    # Some rl-games players also keep hidden states elsewhere (defensive checks)
    if hasattr(agent, "last_states") and agent.last_states is not None:
        try:
            if isinstance(agent.last_states, (list, tuple)):
                agent.last_states = [s.to("cpu") if torch.is_tensor(s) else s for s in agent.last_states]
            elif isinstance(agent.last_states, dict):
                agent.last_states = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in agent.last_states.items()}
            elif torch.is_tensor(agent.last_states):
                agent.last_states = agent.last_states.to("cpu")
            print("[INFO]: Moved agent.last_states to CPU (if present).")
        except Exception:
            pass

    # If rl-games also created any optimizer states on GPU when checkpoint saved, clear or move them:
    try:
        if hasattr(agent, "optimizer") and agent.optimizer is not None:
            for state in agent.optimizer.state.values():
                for k, v in list(state.items()):
                    if torch.is_tensor(v):
                        state[k] = v.cpu()
            print("[INFO]: Moved optimizer state tensors (if present) to CPU.")
    except Exception:
        # non-critical
        pass

    # reset agent internals after model load
    agent.reset()
    if hasattr(agent, "init_rnn") and agent.is_rnn:
        agent.init_rnn()

    # ----------------- Important: make sure obs is on CPU before calling get_action ---------------
    # later in the loop, after obs = agent.obs_to_torch(obs):
    # add an explicit .to("cpu") before agent.get_action(...)


    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            # ensure obs is on CPU (defensive)
            if torch.is_tensor(obs):
                obs = obs.to("cpu")
            elif isinstance(obs, dict):
                obs = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in obs.items()}

            if agent.is_rnn:
                agent.init_rnn()
                if agent.states is not None:
                    agent.states = [s.to("cpu") for s in agent.states]
                print("[INFO]: Reinitialized RNN hidden states on CPU.")

            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
