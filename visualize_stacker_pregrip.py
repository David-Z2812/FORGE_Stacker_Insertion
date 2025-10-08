#!/usr/bin/env python3
"""
Visualize the Stacker Insert environment in a pre-grip configuration.

This script loads the Forge Stacker Insert task, resets to a state where the
stacker is positioned just below the gripper (open), prints pose diagnostics,
and runs a simple viewer loop.
"""

import os

# Prefer CPU rendering/physics to avoid GPU memory issues
os.environ["OMNI_DISABLE_RTX"] = "1"
os.environ["OMNI_DISABLE_DLSS"] = "1"
os.environ["OMNI_DISABLE_RT_RAYTRACING"] = "1"

# Launch Isaac Sim with rendering enabled
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import time

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskStackerInsertCfg


def set_pregrip_configuration(env: ForgeEnv):
    """Place the stacker slightly below the fingertip and open the gripper."""
    # Ensure we have up-to-date fingertip pose
    env.step_sim_no_action()

    # Offset the held asset 4 cm below fingertip, align orientation to fingertip
    offset = torch.tensor([[0.0, 0.0, 0.04]], device=env.device).repeat(env.num_envs, 1)
    target_pos = env.fingertip_midpoint_pos + offset
    target_quat = env.fingertip_midpoint_quat.clone()

    # Write pose to sim
    held_state = env._held_asset.data.default_root_state.clone()
    held_state[:, 0:3] = target_pos + env.scene.env_origins
    held_state[:, 3:7] = target_quat
    held_state[:, 7:] = 0.0
    env._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
    env._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
    env._held_asset.reset()

    # Open gripper DOFs (assumes last 2 joints are gripper)
    if env.ctrl_target_joint_pos is not None and env.ctrl_target_joint_pos.shape[1] >= 9:
        env.ctrl_target_joint_pos[:, 7:] = 0.04  # small opening
        env.step_sim_no_action()


def print_pose_info(env: ForgeEnv, tag: str):
    fingertip_pos = env.fingertip_midpoint_pos[0]
    stacker_pos = env._held_asset.data.root_pos_w[0]
    stacker_quat = env._held_asset.data.root_quat_w[0]
    dist = torch.norm(fingertip_pos - stacker_pos)
    print(f"\n=== {tag} ===")
    print(f"Fingertip pos: {fingertip_pos}")
    print(f"Stacker pos  : {stacker_pos}")
    print(f"Stacker quat : {stacker_quat}")
    print(f"Tip→Stacker  : {dist:.4f} m ({dist*100:.1f} cm)")


def main():
    cfg = ForgeTaskStackerInsertCfg()
    cfg.scene.num_envs = 1
    # CPU-safe scene cloning
    cfg.scene.clone_in_fabric = False
    # Force CPU physics/simulation
    cfg.sim.device = "cpu"

    # Build env
    env = ForgeEnv(cfg, render_mode="human", num_envs=1, device="cpu")

    # Reset and then move to pre-grip
    env.reset()
    # Warm-up renders to ensure viewport initializes
    for _ in range(10):
        env.sim.render()
    print_pose_info(env, "After reset")

    set_pregrip_configuration(env)
    env.step_sim_no_action()
    env.sim.render()
    print_pose_info(env, "Pre-grip configuration")

    print("\nViewer running (no actions applied). Press Ctrl+C to exit.")
    steps = 0
    try:
        while True:
            # Advance simulation without applying policy actions
            env.step_sim_no_action()
            env.sim.render()
            steps += 1
            if steps % 200 == 0:
                print_pose_info(env, f"Tick {steps}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()


