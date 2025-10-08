#!/usr/bin/env python3

"""
Simple pose visualizer that shows the exact poses from debug output.
This creates a minimal scene to visualize the robot and stacker positions.
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
class SimplePoseSceneCfg:
    """Configuration for the simple pose scene."""
    
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
    
    # Stacker configuration
    stacker = RigidObjectCfg(
        prim_path="/World/envs/env_.*/stacker",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/stacker.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

def main():
    """Main function to visualize the poses."""
    
    print("=== SIMPLE POSE VISUALIZER ===\n")
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = SimplePoseSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    print("Setting up poses from debug output...")
    
    # Exact poses from your debug output
    fingertip_pos = torch.tensor([0.5602, 0.7544, 0.7636], device=sim.device)
    stacker_pos = torch.tensor([0.5594, 0.7587, 0.7715], device=sim.device)
    stacker_quat = torch.tensor([0.7071, -0.7071, 0.0000, 0.0000], device=sim.device)
    
    print(f"Fingertip position: {fingertip_pos}")
    print(f"Stacker position: {stacker_pos}")
    print(f"Stacker orientation: {stacker_quat}")
    
    # Calculate distance
    distance = torch.norm(fingertip_pos - stacker_pos)
    print(f"Distance: {distance:.4f}m ({distance * 100:.2f}cm)")
    
    # Position the robot so its fingertip is at the target position
    robot_base_offset = torch.tensor([0.0, 0.0, 0.5], device=sim.device)
    robot_base_pos = fingertip_pos - robot_base_offset
    
    # Set robot position
    robot_entity = scene["robot"]
    robot_entity.write_root_pose_to_sim(robot_base_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device))
    
    # Set stacker position
    stacker_entity = scene["stacker"]
    stacker_entity.write_root_pose_to_sim(stacker_pos, stacker_quat)
    
    print(f"Robot base position: {robot_base_pos}")
    print(f"Stacker position: {stacker_pos}")
    
    # Run simulation
    print("\nStarting simulation...")
    print("Press Ctrl+C to exit")
    
    try:
        step_count = 0
        while True:
            # Step simulation
            sim.step()
            
            # Update scene
            scene.update(sim.get_physics_dt())
            
            # Print current positions every 60 steps (1 second)
            if step_count % 60 == 0:
                current_stacker = stacker_entity.data.root_pos_w[0]
                current_robot = robot_entity.data.root_pos_w[0]
                current_distance = torch.norm(current_stacker - current_robot)
                print(f"Step {step_count}: Distance={current_distance:.4f}m")
                
            step_count += 1
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Cleanup
    sim.close()

if __name__ == "__main__":
    main()
