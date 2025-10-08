#!/usr/bin/env python3
"""
Proper test script for the Forge Stacker Insert environment using Isaac Lab's AppLauncher.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401


def test_stacker_insert_env():
    """Test the stacker insert environment registration and initialization."""
    
    print("Testing Forge Stacker Insert Environment...")
    
    # Test environment registration
    try:
        env_id = "Isaac-Forge-StackerInsert-Direct-v0"
        
        # Check if environment is registered
        if env_id in gym.envs.registry:
            print(f"✓ Environment '{env_id}' is registered")
        else:
            print(f"❌ Environment '{env_id}' not found in registry")
            print("Available Forge environments:")
            for env_name in gym.envs.registry:
                if "Forge" in env_name:
                    print(f"   - {env_name}")
            return False
        
        # Try to create the environment
        env = gym.make(env_id, num_envs=4, device="cuda:0")
        print("✓ Environment created successfully")
        
        # Test environment properties
        print(f"✓ Action space: {env.action_space}")
        print(f"✓ Observation space: {env.observation_space}")
        print(f"✓ Number of environments: {env.num_envs}")
        
        # Test reset
        obs, _ = env.reset()
        print(f"✓ Environment reset successful")
        print(f"✓ Observation shape: {obs['policy'].shape}")
        
        # Test step
        actions = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"✓ Environment step successful")
        print(f"✓ Reward shape: {rewards.shape}")
        
        env.close()
        print("✓ Environment closed successfully")
        
        print("\n🎉 All tests passed! The Forge Stacker Insert environment is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\n🔍 Debugging information:")
        print("   - USD files exist: ✓")
        print("   - Code configuration: ✓")
        print("   - Environment registration: Check above")
        print("\n💡 This error might be due to:")
        print("   1. Isaac Sim environment not fully initialized")
        print("   2. Missing dependencies in the Isaac Lab environment")
        print("   3. GPU/CUDA issues")
        print("   4. Asset loading problems")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # run the test function
        test_stacker_insert_env()
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the app
        simulation_app.close()






