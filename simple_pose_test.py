#!/usr/bin/env python3

"""
Simple pose test script that follows the same pattern as existing environments.
This creates a minimal scene to test the poses.
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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, UsdFileCfg

@configclass
class SimplePoseTestCfg:
    """Configuration for the simple pose test."""
    
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

class SimplePoseTest(DirectRLEnv):
    """Simple pose test environment."""
    
    cfg: SimplePoseTestCfg
    
    def __init__(self, cfg: SimplePoseTestCfg, render_mode: str | None = None, **kwargs):
        # Set up basic configuration
        cfg.observation_space = 0
        cfg.state_space = 0
        cfg.action_space = 0
        
        super().__init__(cfg, render_mode, **kwargs)
    
    def _setup_scene(self):
        """Set up the scene."""
        # Create simulation context
        self.sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, substeps=1)
        self.sim = sim_utils.SimulationContext(self.sim_cfg)
        
        # Create robot
        self.robot = Articulation(self.cfg.robot)
        
        # Create stacker
        self.stacker = RigidObject(self.cfg.stacker)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Register objects with scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["stacker"] = self.stacker
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _get_observations(self):
        """Get observations."""
        return {}
    
    def _get_rewards(self):
        """Get rewards."""
        return torch.zeros(self.num_envs, device=self.device)
    
    def _get_dones(self):
        """Get done flags."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def _get_info(self):
        """Get info."""
        return {}
    
    def _apply_action(self, action):
        """Apply action."""
        pass

def main():
    """Main function to test the poses."""
    
    print("=== SIMPLE POSE TEST ===\n")
    
    # Create environment
    cfg = SimplePoseTestCfg()
    env = SimplePoseTest(cfg)
    
    print("Setting up poses from debug output...")
    
    # Exact poses from your debug output
    fingertip_pos = torch.tensor([0.5602, 0.7544, 0.7636], device=env.device)
    stacker_pos = torch.tensor([0.5594, 0.7587, 0.7715], device=env.device)
    stacker_quat = torch.tensor([0.7071, -0.7071, 0.0000, 0.0000], device=env.device)
    
    print(f"Fingertip position: {fingertip_pos}")
    print(f"Stacker position: {stacker_pos}")
    print(f"Stacker orientation: {stacker_quat}")
    
    # Calculate distance
    distance = torch.norm(fingertip_pos - stacker_pos)
    print(f"Distance: {distance:.4f}m ({distance * 100:.2f}cm)")
    
    # Position the robot so its fingertip is at the target position
    robot_base_offset = torch.tensor([0.0, 0.0, 0.5], device=env.device)
    robot_base_pos = fingertip_pos - robot_base_offset
    
    # Set robot position
    env.robot.write_root_pose_to_sim(robot_base_pos, torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device))
    
    # Set stacker position
    env.stacker.write_root_pose_to_sim(stacker_pos, stacker_quat)
    
    print(f"Robot base position: {robot_base_pos}")
    print(f"Stacker position: {stacker_pos}")
    
    # Run simulation
    print("\nStarting simulation...")
    print("Press Ctrl+C to exit")
    
    try:
        step_count = 0
        while True:
            # Step simulation
            env.sim.step()
            
            # Update scene
            env.scene.update(env.sim.get_physics_dt())
            
            # Print current positions every 60 steps (1 second)
            if step_count % 60 == 0:
                current_stacker = env.stacker.data.root_pos_w[0]
                current_robot = env.robot.data.root_pos_w[0]
                current_distance = torch.norm(current_stacker - current_robot)
                print(f"Step {step_count}: Distance={current_distance:.4f}m")
                
            step_count += 1
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Cleanup
    env.sim.close()

if __name__ == "__main__":
    main()






