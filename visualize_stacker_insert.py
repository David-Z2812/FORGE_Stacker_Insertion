#!/usr/bin/env python3
"""
Visualization script for the Forge Stacker Insert environment.
This script loads the environment in Isaac Sim with visualization enabled.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app with rendering enabled
app_launcher = AppLauncher(headless=False)  # Set to False for visualization
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskStackerInsertCfg


def visualize_stacker_insert():
    """Visualize the stacker insert environment in Isaac Sim."""
    
    print("🎬 Loading Forge Stacker Insert Environment for Visualization...")
    
    try:
        # Create the configuration
        cfg = ForgeTaskStackerInsertCfg()
        cfg.scene.num_envs = 1  # Just one environment for testing
        print("✓ Configuration created successfully")
        
        # Create the environment with visualization
        env = ForgeEnv(cfg, render_mode="human", num_envs=1, device="cpu")
        print("✓ Environment created successfully with visualization")
        
        # Show pose information
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
        
        # Reset the environment
        obs, _ = env.reset()
        print("✓ Environment reset - you should see the scene in Isaac Sim")
        
        # Show poses after reset
        print("\n=== POSES AFTER RESET ===")
        fingertip_pos_reset = env.fingertip_midpoint_pos[0]
        stacker_pos_reset = env._held_asset.data.root_pos_w[0]
        stacker_quat_reset = env._held_asset.data.root_quat_w[0]
        
        print(f"Fingertip position: {fingertip_pos_reset}")
        print(f"Stacker position: {stacker_pos_reset}")
        print(f"Stacker orientation: {stacker_quat_reset}")
        
        distance_reset = torch.norm(fingertip_pos_reset - stacker_pos_reset)
        print(f"Distance: {distance_reset:.4f}m ({distance_reset * 100:.2f}cm)")
        
        if stacker_pos_reset[2] > fingertip_pos_reset[2]:
            print("❌ Stacker is ABOVE fingertip")
        else:
            print("✅ Stacker is BELOW fingertip")
        
        print("=== END POSES AFTER RESET ===\n")
        
        print("\n🎮 Controls:")
        print("   - The environment is now running in Isaac Sim")
        print("   - You can see the stacker and container corner casting")
        print("   - Press Ctrl+C to exit")
        print("   - The simulation will run automatically")
        
        # Run the simulation for a while
        step_count = 0
        max_steps = 1000  # Run for 1000 steps (about 10 seconds at 100Hz)
        
        while step_count < max_steps:
            # Take random actions
            actions = torch.randn((env.num_envs, env.action_space.shape[0]), device=env.device) * 0.1
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}/{max_steps} - Reward: {rewards.item():.4f}")
            
            # Reset if episode ended
            if terminated or truncated:
                obs, _ = env.reset()
                print(f"Episode ended at step {step_count}, resetting...")
        
        print(f"\n✅ Visualization complete! Ran for {step_count} steps.")
        print("   You should have seen the stacker insertion environment in Isaac Sim.")
        
        env.close()
        print("✓ Environment closed successfully")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
        if 'env' in locals():
            env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # run the visualization
        visualize_stacker_insert()
    except Exception as e:
        print(f"Error running visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the app
        simulation_app.close()






