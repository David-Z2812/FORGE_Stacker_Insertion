#!/usr/bin/env python3
"""
Simple visualization script for the Forge Stacker Insert environment.
This script loads the environment with contact sensors disabled for visualization.
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


def visualize_stacker_insert_simple():
    """Visualize the stacker insert environment in Isaac Sim with contact sensors disabled."""
    
    print("🎬 Loading Forge Stacker Insert Environment for Simple Visualization...")
    
    try:
        # Create the configuration
        cfg = ForgeTaskStackerInsertCfg()
        print("✓ Configuration created successfully")
        
        # Disable contact sensors for visualization
        cfg.task.fixed_asset.spawn.activate_contact_sensors = False
        cfg.task.held_asset.spawn.activate_contact_sensors = False
        print("✓ Contact sensors disabled for visualization")
        
        # Create the environment with visualization
        env = ForgeEnv(cfg, render_mode="human", num_envs=1, device="cuda:0")
        print("✓ Environment created successfully with visualization")
        
        # Reset the environment
        obs, _ = env.reset()
        print("✓ Environment reset - you should see the scene in Isaac Sim")
        
        print("\n🎮 Isaac Sim Visualization:")
        print("   - The environment is now running in Isaac Sim")
        print("   - You can see the stacker and container corner casting")
        print("   - The robot arm should be visible")
        print("   - Press Ctrl+C to exit")
        print("   - The simulation will run automatically")
        
        # Run the simulation for a while
        step_count = 0
        max_steps = 500  # Run for 500 steps (about 5 seconds at 100Hz)
        
        while step_count < max_steps:
            # Take small random actions
            actions = torch.randn((env.num_envs, env.action_space.shape[0]), device=env.device) * 0.05
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
        visualize_stacker_insert_simple()
    except Exception as e:
        print(f"Error running visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the app
        simulation_app.close()






