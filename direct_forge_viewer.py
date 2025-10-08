#!/usr/bin/env python3
"""
Direct Forge Stacker Insert environment viewer (bypasses gym registration).
"""

import os
import time
import signal
import sys


from isaaclab.app import AppLauncher

# --- CPU-Konfiguration ---
CONFIG = {
    "headless": False,       # Fenster anzeigen
    "enable_gpu": False,     # keine GPU-Nutzung
    "device": "cpu",         # zwingt IsaacLab auf CPU
}
app_launcher = AppLauncher(CONFIG)
simulation_app = app_launcher.app

# Import the environment and configuration directly
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg
import torch

# Global flag to control the main loop
running = True

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n⏹️  Stopping environment...")
    running = False

def main():
    print("🎬 Loading Forge Stacker Insert Environment (Direct Mode)...")
    
    try:
        # Create the environment configuration
        print("Creating configuration...")
        cfg = ForgeTaskPegInsertCfg()
        print("✅ Configuration created successfully")
        
        # Print some config details
        print(f"   - Task name: {cfg.task_name}")
        print(f"   - Episode length: {cfg.episode_length_s}s")
        print(f"   - Number of environments: 1")
        
        # Create the environment directly
        print("Creating environment...")
        env = ForgeEnv(
            cfg=cfg,
            render_mode="human", 
            num_envs=1, 
            device="cpu"
        )
        print("✅ Environment created successfully")
        
        # Reset the environment
        print("Resetting environment...")
        obs, _ = env.reset()
        print("✅ Environment reset successful")
        
        print("\n🎮 Isaac Sim Environment Visualization:")
        print("   - You should see the complete Forge environment")
        print("   - Robot arm with stacker in hand")
        print("   - Container corner casting on the table")
        print("   - Use mouse to rotate, zoom, and pan the view")
        print("   - Press Ctrl+C to exit")
        
        # Keep the environment running
        global running
        step_count = 0
        while running:
            # Take a zero action to keep the environment active
            actions = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            step_count += 1
            if step_count % 100 == 0:
                print(f"   Step {step_count} - Environment running...")
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
        
        print("Closing environment...")
        env.close()
        print("✅ Environment closed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\n🔍 This might be due to:")
        print("   1. USD assets not found")
        print("   2. Environment configuration issues")
        print("   3. GPU memory problems")
        print("   4. Missing dependencies")
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

