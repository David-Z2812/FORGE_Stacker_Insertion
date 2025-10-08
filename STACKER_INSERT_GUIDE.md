# Forge Stacker Insert Environment Setup Guide

## Overview

This guide explains how to modify the Isaac Lab Forge framework from peg insertion to inserting a stacker into a container corner casting. The modifications have been implemented and are ready for use once the required assets are created.

## What Has Been Modified

### 1. **New Task Configuration** (`forge_tasks_cfg.py`)

Added three new classes:

- **`Stacker`**: Configuration for the held asset (the stacker being inserted)
- **`ContainerCornerCasting`**: Configuration for the fixed asset (the target receptacle)
- **`ForgeStackerInsert`**: Main task configuration combining both assets

### 2. **Environment Configuration** (`forge_env_cfg.py`)

Added:
- **`ForgeTaskStackerInsertCfg`**: Environment configuration for the stacker insert task

### 3. **Environment Registration** (`__init__.py`)

Added:
- **`Isaac-Forge-StackerInsert-Direct-v0`**: Gym environment registration

## Key Parameters for Stacker Insertion

### Asset Specifications
- **Stacker**: 50mm diameter, 80mm height, 100g mass
- **Container Corner Casting**: 60mm insertion point, 100mm height, 200g mass
- **Episode Duration**: 15 seconds (longer than peg insertion for more complex task)

### Robot Initialization
- **Hand Position**: 60mm above the insertion point
- **Position Noise**: ±30mm in X/Y, ±20mm in Z
- **Orientation**: Upright with ±45° yaw noise

### Reward Configuration
- **Success Threshold**: 5% of insertion depth
- **Contact Penalty**: 0.15 (moderate penalty for excessive force)
- **Keypoint Scaling**: Optimized for larger objects

## Required Assets ✅ COMPLETED

You have successfully created the two USD assets:

### 1. **Stacker Asset** (`factory_stacker.usd`) ✅
- **Location**: `/home/david/IsaacLab/source/isaaclab_assets/data/Factory/factory_stacker.usd`
- **Dimensions**: ~50mm diameter × 80mm height
- **Shape**: Cylindrical or rectangular stacker with appropriate geometry
- **Materials**: Standard factory materials with proper collision geometry

### 2. **Container Corner Casting** (`factory_container_corner_casting.usd`) ✅
- **Location**: `/home/david/IsaacLab/source/isaaclab_assets/data/Factory/factory_container_corner_casting.usd`
- **Dimensions**: ~60mm insertion point, 100mm height
- **Shape**: Corner casting with insertion receptacle
- **Materials**: Standard factory materials with proper collision geometry

## How to Create the Assets

### Option 1: Use Isaac Sim's Asset Creation Tools
1. Open Isaac Sim
2. Create new USD files for both assets
3. Design the geometry using Isaac Sim's modeling tools
4. Add proper collision geometry
5. Set appropriate materials and physics properties
6. Save to the Factory directory in the nucleus assets

### Option 2: Import from CAD Software
1. Create models in your preferred CAD software (SolidWorks, Fusion 360, etc.)
2. Export as USD format
3. Import into Isaac Sim
4. Add physics properties and collision geometry
5. Save to the appropriate location

### Option 3: Modify Existing Assets
1. Copy existing factory assets (e.g., `factory_peg_8mm.usd`)
2. Modify the geometry to match your stacker and corner casting
3. Adjust dimensions and materials as needed

## Testing the Environment ✅ WORKING

The environment is now successfully working! Here's how to test it:

### **Direct Environment Creation (Recommended)**
```bash
cd /home/david/IsaacLab
./isaaclab.sh -p direct_test.py
```

### **Gymnasium Environment (Alternative)**
```bash
cd /home/david/IsaacLab
./isaaclab.sh -p proper_test.py
```

**Status**: ✅ **Environment is working correctly!**
- ✅ Configuration loads successfully
- ✅ USD assets load correctly
- ✅ Environment creates and runs
- ✅ Basic functionality works

**Note**: You may see warnings about contact sensors and rigid body properties. These are common with custom USD assets and don't prevent the environment from working.

## Using the Environment

### Basic Usage
```python
import gymnasium as gym

# Create the environment
env = gym.make("Isaac-Forge-StackerInsert-Direct-v0", num_envs=128, device="cuda:0")

# Reset and run
obs, _ = env.reset()
for _ in range(1000):
    actions = env.action_space.sample()  # Random actions for testing
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    if terminated.any() or truncated.any():
        obs, _ = env.reset()

env.close()
```

### Training with RL Games
```python
# Use the existing RL Games configuration
from isaaclab_tasks.direct.forge.agents import rl_games_ppo_cfg

# The environment will automatically use the PPO configuration
```

## Key Differences from Peg Insertion

1. **Larger Objects**: Stacker and corner casting are significantly larger than peg/hole
2. **More Complex Geometry**: Corner casting has more complex insertion geometry
3. **Longer Episodes**: 15 seconds vs 10 seconds for peg insertion
4. **Adjusted Parameters**: Noise levels, success thresholds, and contact penalties optimized for larger objects
5. **Different Success Criteria**: Based on insertion depth rather than simple hole filling

## Troubleshooting

### Common Issues

1. **Asset Not Found**: Ensure USD files are in the correct location
2. **Collision Issues**: Check that collision geometry is properly defined
3. **Physics Instability**: Adjust mass, friction, and damping parameters
4. **Reward Issues**: Tune success thresholds and keypoint coefficients

### Debugging Tips

1. Use Isaac Sim's visualization to check asset placement
2. Monitor force/torque sensors for contact issues
3. Adjust episode length if task is too easy/hard
4. Fine-tune reward parameters based on training performance

## Next Steps

1. ✅ **Create the USD assets** - COMPLETED
2. ✅ **Update the code configuration** - COMPLETED  
3. ✅ **Test the environment** - COMPLETED
4. **Train a policy** using your preferred RL algorithm
5. **Fine-tune parameters** based on training results
6. **Deploy and evaluate** the trained policy

## Current Status: 🎉 **SUCCESS!**

Your Forge Stacker Insert environment is now fully functional! The environment:
- ✅ Loads your custom USD assets correctly
- ✅ Creates and runs without errors
- ✅ Is ready for training

You can now proceed with training a reinforcement learning policy for the stacker insertion task.

## Additional Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim USD Tutorial](https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_sim_tutorials/tutorial_creating_scenes.html)
- [Forge Framework Paper](https://arxiv.org/abs/2408.04587)

---

**Note**: The code modifications are complete and ready to use. The main remaining task is creating the USD assets for the stacker and container corner casting.
