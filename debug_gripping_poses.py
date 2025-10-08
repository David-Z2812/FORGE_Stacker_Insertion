#!/usr/bin/env python3

"""
Debug script to visualize the exact poses from the gripping failure.
This loads the robot and stacker in the positions shown in the debug output.
"""

import torch
import numpy as np
from isaaclab import sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, UsdFileCfg
from isaaclab.utils.dict import update_dict
from isaaclab.utils import torch_utils

@configclass
class DebugSceneCfg:
    """Configuration for the debug scene."""
    
    # Robot configuration
    robot = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/franka_mimic.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
                enable_gyroscopic_forces=True,
                disable_gravity=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
                max_contact_impulse=100000.0,
            ),
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

def main():
    """Main function to create and visualize the debug scene."""
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, substeps=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Create scene
    scene_cfg = DebugSceneCfg()
    scene = InteractiveScene(scene_cfg)
    
    # Set up the exact poses from debug output
    print("Setting up exact poses from debug output...")
    
    # Robot fingertip position from debug output
    fingertip_pos = torch.tensor([0.5602, 0.7544, 0.7636], device=sim.device)
    
    # Stacker position from debug output  
    stacker_pos = torch.tensor([0.5594, 0.7587, 0.7715], device=sim.device)
    
    # Stacker orientation from debug output
    stacker_quat = torch.tensor([0.7071, -0.7071, 0.0000, 0.0000], device=sim.device)
    
    # Set robot position (approximate - you may need to adjust this)
    # The robot base should be positioned so the fingertip reaches the target position
    robot_base_pos = fingertip_pos - torch.tensor([0.0, 0.0, 0.5], device=sim.device)  # Approximate offset
    
    # Set robot state
    robot_entity = scene["robot"]
    robot_entity.write_root_pose_to_sim(robot_base_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=sim.device))
    
    # Set stacker state
    stacker_entity = scene["stacker"]
    stacker_entity.write_root_pose_to_sim(stacker_pos, stacker_quat)
    
    print(f"Robot base position: {robot_base_pos}")
    print(f"Fingertip position: {fingertip_pos}")
    print(f"Stacker position: {stacker_pos}")
    print(f"Stacker orientation: {stacker_quat}")
    
    # Calculate distance between fingertip and stacker
    distance = torch.norm(fingertip_pos - stacker_pos)
    print(f"Distance between fingertip and stacker: {distance:.4f} meters")
    
    # Check if they're close enough for gripping
    gripping_threshold = 0.05  # 5cm
    if distance < gripping_threshold:
        print("✅ Objects are close enough for gripping")
    else:
        print(f"❌ Objects are too far apart for gripping (threshold: {gripping_threshold}m)")
    
    # Run simulation
    print("\nStarting simulation...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Step simulation
            sim.step()
            
            # Update scene
            scene.update(sim.get_physics_dt())
            
            # Print current positions every 60 steps (1 second)
            if sim.get_physics_step_count() % 60 == 0:
                current_fingertip = robot_entity.data.root_pos_w[0]  # This might need adjustment
                current_stacker = stacker_entity.data.root_pos_w[0]
                current_distance = torch.norm(current_fingertip - current_stacker)
                print(f"Step {sim.get_physics_step_count()}: Distance = {current_distance:.4f}m")
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Cleanup
    sim.close()

if __name__ == "__main__":
    main()

