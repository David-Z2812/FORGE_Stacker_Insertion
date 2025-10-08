#!/usr/bin/env python3

"""
Simple debug script to visualize the exact poses from the gripping failure.
This creates a minimal scene with just the robot and stacker in the exact positions.
"""

import torch
import numpy as np
from isaaclab import sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, UsdFileCfg

@configclass
class SimpleDebugSceneCfg:
    """Configuration for the simple debug scene."""
    
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
    """Main function to create and visualize the debug scene."""
    
    print("Creating simulation context...")
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    print("Creating scene...")
    
    # Create scene
    scene_cfg = SimpleDebugSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    print("Setting up exact poses from debug output...")
    
    # Exact poses from your debug output
    fingertip_pos = torch.tensor([0.5602, 0.7544, 0.7636], device=sim.device)
    stacker_pos = torch.tensor([0.5594, 0.7587, 0.7715], device=sim.device)
    stacker_quat = torch.tensor([0.7071, -0.7071, 0.0000, 0.0000], device=sim.device)
    
    # Calculate distance
    distance = torch.norm(fingertip_pos - stacker_pos)
    print(f"Distance between fingertip and stacker: {distance:.4f} meters")
    
    # Check gripping feasibility
    gripping_threshold = 0.05  # 5cm
    if distance < gripping_threshold:
        print("✅ Objects are close enough for gripping")
    else:
        print(f"❌ Objects are too far apart for gripping (threshold: {gripping_threshold}m)")
    
    # Position the robot so its fingertip is at the target position
    # This is an approximation - you may need to adjust
    robot_base_offset = torch.tensor([0.0, 0.0, 0.5], device=sim.device)
    robot_base_pos = fingertip_pos - robot_base_offset
    
    # Set robot position
    robot_entity = scene["robot"]
    robot_entity.write_root_pose_to_sim(robot_base_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device))
    
    # Set stacker position
    stacker_entity = scene["stacker"]
    stacker_entity.write_root_pose_to_sim(stacker_pos, stacker_quat)
    
    print(f"Robot base position: {robot_base_pos}")
    print(f"Target fingertip position: {fingertip_pos}")
    print(f"Stacker position: {stacker_pos}")
    print(f"Stacker orientation: {stacker_quat}")
    
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
                print(f"Step {step_count}: Stacker position = {current_stacker}")
                
            step_count += 1
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Cleanup
    sim.close()

if __name__ == "__main__":
    main()

