#!/usr/bin/env python3

"""
Test script that uses the existing Forge environment to show the poses.
This is the simplest approach that should work with your existing setup.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app with rendering enabled
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskStackerInsertCfg

def main():
    """Main function to test the poses using the existing Forge environment."""
    
    print("=== TESTING FORGE POSES ===\n")
    
    # Create environment
    cfg = ForgeTaskStackerInsertCfg()
    cfg.scene.num_envs = 1  # Just one environment for testing
    cfg.scene.env_spacing = 1.0
    
    env = ForgeEnv(cfg, device="cpu")
    
    print("Environment created successfully!")
    print("Setting up poses from debug output...")
    
    # Get the current poses
    fingertip_pos = env.fingertip_midpoint_pos[0]
    stacker_pos = env._held_asset.data.root_pos_w[0]
    stacker_quat = env._held_asset.data.root_quat_w[0]
    
    print(f"Fingertip position: {fingertip_pos}")
    print(f"Stacker position: {stacker_pos}")
    print(f"Stacker orientation: {stacker_quat}")
    
    # Calculate distance
    distance = torch.norm(fingertip_pos - stacker_pos)
    print(f"Distance: {distance:.4f}m ({distance * 100:.2f}cm)")
    
    # Check if stacker is above or below fingertip
    if stacker_pos[2] > fingertip_pos[2]:
        print("❌ Stacker is ABOVE fingertip")
    else:
        print("✅ Stacker is BELOW fingertip")
    
    # Check gripping feasibility
    gripping_threshold = 0.05  # 5cm
    if distance < gripping_threshold:
        print("✅ Objects are close enough for gripping")
    else:
        print(f"❌ Objects are too far apart for gripping (threshold: {gripping_threshold}m)")
    
    # Run simulation for a few steps to see the poses
    print("\nRunning simulation for 10 steps...")
    
    for step in range(10):
        # Step the environment
        obs, reward, done, info = env.step(torch.zeros(1, 7, device=env.device))
        
        # Get current poses
        current_fingertip = env.fingertip_midpoint_pos[0]
        current_stacker = env._held_asset.data.root_pos_w[0]
        current_distance = torch.norm(current_fingertip - current_stacker)
        
        print(f"Step {step}: Distance={current_distance:.4f}m")
        
        if step == 0:
            print(f"  Fingertip: {current_fingertip}")
            print(f"  Stacker: {current_stacker}")
    
    print("\nSimulation completed!")
    print("You should see the robot and stacker in the 3D view.")
    print("Press Ctrl+C to exit the viewer.")
    
    # Keep the viewer open
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    main()
