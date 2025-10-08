#!/usr/bin/env python3
"""
Direct test script for the Forge Stacker Insert environment.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskStackerInsertCfg


def test_direct_creation():
    """Test creating the environment directly."""
    
    print("Testing Direct Forge Stacker Insert Environment Creation...")
    
    try:
        # Create the configuration
        cfg = ForgeTaskStackerInsertCfg()
        print("✓ Configuration created successfully")
        
        # Create the environment directly
        env = ForgeEnv(cfg, render_mode=None, num_envs=4, device="cuda:0")
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
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # run the test function
        test_direct_creation()
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the app
        simulation_app.close()






