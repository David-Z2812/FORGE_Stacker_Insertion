#!/usr/bin/env python3
"""
View the complete Forge Stacker Insert environment (fixed version).
"""

import os
import time
import signal
import sys

# Set environment variables to disable RTX and use basic rendering
os.environ["OMNI_DISABLE_RTX"] = "1"
os.environ["OMNI_DISABLE_DLSS"] = "1"
os.environ["OMNI_DISABLE_RT_RAYTRACING"] = "1"

from isaaclab.app import AppLauncher

# Launch Isaac Sim with basic rendering
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskStackerInsertCfg
import torch

# Global flag to control the main loop
running = True

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n⏹️  Stopping environment...")
    running = False

def main():
    print("🎬 Loading Forge Stacker Insert Environment...")
    
    try:
        # Create the environment configuration
        cfg = ForgeTaskStackerInsertCfg()
        print("✅ Configuration created successfully")
        
        # Create the environment directly (not through gym.make)
        env = ForgeEnv(cfg, render_mode="human", num_envs=1, device="cuda:0")
        print("✅ Environment created successfully")
        
        # Reset the environment
        obs, _ = env.reset()
        print("✅ Environment reset successful")
        
        print("🎮 Isaac Sim Environment Visualization:")
        print("   - You should see the complete Forge environment")
        print("   - Robot arm with stacker in hand")
        print("   - Container corner casting on the table")
        print("   - Use mouse to rotate, zoom, and pan the view")
        print("   - Press Ctrl+C to exit")
        
        # Keep the environment running
        global running
        step_count = 0
        while running:
            # Take a random action to keep the environment active
            actions = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"   Step {step_count} - Environment running...")
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
        
        env.close()
        print("✅ Environment closed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\n🔍 This might be due to:")
        print("   1. USD assets not found")
        print("   2. Environment configuration issues")
        print("   3. GPU memory problems")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping environment...")
    finally:
        simulation_app.close()




