#!/usr/bin/env python3

"""
Visualize the poses with the fix applied.
This shows both the original problematic poses and the fixed poses.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app with rendering enabled
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, UsdFileCfg

@configclass
class FixedPoseSceneCfg:
    """Configuration for the fixed pose scene."""
    
    # Robot configuration
    robot = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/franka_mimic.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,
                "panda_joint7": 0.785,
                "panda_finger_joint1": 0.0,
                "panda_finger_joint2": 0.0,
            },
        ),
    )
    
    # Original stacker (problematic)
    stacker_original = RigidObjectCfg(
        prim_path="/World/envs/env_.*/stacker_original",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/stacker.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    
    # Fixed stacker
    stacker_fixed = RigidObjectCfg(
        prim_path="/World/envs/env_.*/stacker_fixed",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/stacker.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

def main():
    """Main function to visualize the fixed poses."""
    
    print("=== FIXED POSE VISUALIZER ===\n")
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = FixedPoseSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    print("Setting up poses...")
    
    # Fingertip position from debug output
    fingertip_pos = torch.tensor([0.5602, 0.7544, 0.7636], device=sim.device)
    
    # Original problematic poses
    stacker_original_pos = torch.tensor([0.5594, 0.7587, 0.7715], device=sim.device)
    stacker_original_quat = torch.tensor([0.7071, -0.7071, 0.0000, 0.0000], device=sim.device)
    
    # Fixed poses (using the fix we applied)
    relative_pos_fixed = torch.tensor([-0.0008, 0.0043, -0.02], device=sim.device)  # 2cm below
    stacker_fixed_pos = fingertip_pos + relative_pos_fixed
    stacker_fixed_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device)  # Same as gripper
    
    print(f"Fingertip position: {fingertip_pos}")
    print(f"Original stacker position: {stacker_original_pos}")
    print(f"Fixed stacker position: {stacker_fixed_pos}")
    
    # Calculate distances
    distance_original = torch.norm(fingertip_pos - stacker_original_pos)
    distance_fixed = torch.norm(fingertip_pos - stacker_fixed_pos)
    
    print(f"Original distance: {distance_original:.4f}m ({distance_original * 100:.2f}cm)")
    print(f"Fixed distance: {distance_fixed:.4f}m ({distance_fixed * 100:.2f}cm)")
    
    # Position the robot
    robot_base_offset = torch.tensor([0.0, 0.0, 0.5], device=sim.device)
    robot_base_pos = fingertip_pos - robot_base_offset
    
    # Set robot position
    robot_entity = scene["robot"]
    robot_entity.write_root_pose_to_sim(robot_base_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device))
    
    # Set original stacker position
    stacker_original_entity = scene["stacker_original"]
    stacker_original_entity.write_root_pose_to_sim(stacker_original_pos, stacker_original_quat)
    
    # Set fixed stacker position
    stacker_fixed_entity = scene["stacker_fixed"]
    stacker_fixed_entity.write_root_pose_to_sim(stacker_fixed_pos, stacker_fixed_quat)
    
    print(f"Robot base position: {robot_base_pos}")
    print(f"Original stacker position: {stacker_original_pos}")
    print(f"Fixed stacker position: {stacker_fixed_pos}")
    
    # Run simulation
    print("\nStarting simulation...")
    print("Press Ctrl+C to exit")
    print("You should see:")
    print("- Robot in the scene")
    print("- Original stacker (problematic position)")
    print("- Fixed stacker (correct position below gripper)")
    
    try:
        step_count = 0
        while True:
            # Step simulation
            sim.step()
            
            # Update scene
            scene.update(sim.get_physics_dt())
            
            # Print current positions every 60 steps (1 second)
            if step_count % 60 == 0:
                current_original = stacker_original_entity.data.root_pos_w[0]
                current_fixed = stacker_fixed_entity.data.root_pos_w[0]
                current_robot = robot_entity.data.root_pos_w[0]
                
                dist_orig = torch.norm(current_original - current_robot)
                dist_fixed = torch.norm(current_fixed - current_robot)
                
                print(f"Step {step_count}: Original distance={dist_orig:.4f}m, Fixed distance={dist_fixed:.4f}m")
                
            step_count += 1
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Cleanup
    sim.close()

if __name__ == "__main__":
    main()
