#!/usr/bin/env python3

"""
CPU-only visualization script for the Forge Stacker Insert environment.
This script is optimized for CPU usage and should work without CUDA.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app with rendering enabled
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskStackerInsertCfg

def main():
    """Main function for CPU-only visualization."""
    
    print("=== CPU-ONLY FORGE POSE VISUALIZATION ===\n")
    
    try:
        # Create the configuration
        cfg = ForgeTaskStackerInsertCfg()
        cfg.scene.num_envs = 1  # Just one environment for testing
        cfg.scene.env_spacing = 1.0
        
        # Force CPU usage
        cfg.sim.device = "cpu"
        cfg.sim.use_gpu = False
        
        print("✓ Configuration created for CPU usage")
        
        # Create the environment with CPU
        env = ForgeEnv(cfg, render_mode="human", num_envs=1, device="cpu")
        print("✓ Environment created successfully with CPU")
        
        # Reset the environment first to initialize all attributes
        obs, _ = env.reset()
        print("✓ Environment reset - you should see the scene in Isaac Sim")
        
        # Show pose information after reset
        print("\n=== POSE INFORMATION ===")
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
        
        print("=== END POSE INFORMATION ===\n")
        
        print("\n🎮 Controls:")
        print("   - The environment is now running in Isaac Sim with CPU")
        print("   - You can see the stacker and robot in the 3D view")
        print("   - Press Ctrl+C to exit")
        
        # Keep the viewer open
        try:
            while True:
                import time
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nExiting...")
        
        # Cleanup
        env.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Isaac Sim is properly installed and configured for CPU usage.")

if __name__ == "__main__":
    main()