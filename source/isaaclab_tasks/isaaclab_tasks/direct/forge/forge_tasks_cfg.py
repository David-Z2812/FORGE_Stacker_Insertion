# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import (
    FactoryTask, 
    FixedAssetCfg, 
    GearMesh, 
    HeldAssetCfg, 
    NutThread, 
    PegInsert
)
from pathlib import Path

# Define local asset directory for your USD files
LOCAL_ASSET_DIR = str(Path(__file__).resolve().parents[4] / "isaaclab_assets/data/Factory/")


@configclass
class ForgeTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class Stacker(HeldAssetCfg):
    usd_path = f"{LOCAL_ASSET_DIR}/stacker_oriented.usd"
    diameter = 0.110  # Y-dimension (width for gripper) 0.137
    height = 0.110    # Z-dimension (height)
    mass = 0.1        # Mass of the stacker
    friction = 0.75


@configclass
class ContainerCornerCasting(FixedAssetCfg):
    usd_path = f"{LOCAL_ASSET_DIR}/c_corner_positioned_3.usd"
    diameter = 0.142  # X-dimension (insertion point width)
    # height = 0.118    # Z-dimension (height)
    height = 0.0    # Z-dimension (height)
    base_height = 0.06799  # Stacker_lowest point to container lowest point difference during insertion
    friction = 0.75
    mass = 10.0


@configclass
class ForgeStackerInsert(ForgeTask):
    name = "stacker_insert"
    fixed_asset_cfg = ContainerCornerCasting()
    held_asset_cfg = Stacker()
    asset_size = 120.0  # Size in mm
    duration_s = 10.0  # duration of each episode

    # Robot initialization
    hand_init_pos: list = [0.00, 0.00, -0.105]  # Relative to fixed asset tip
    hand_init_pos_noise: list = [0.03, 0.03, 0.02]  # Slightly more noise for larger objects
    # hand_init_pos_noise: list = [0.00, 0.00, 0.0]
    hand_init_orn: list = [0.0, 0.0, -1.571-0.524]
    hand_init_orn_noise: list = [0.2, 0.2, 0.2]  # No orientation noise for deterministic behavior
    # hand_init_orn_noise: list = [0.0, 0.0, 0.0]

    # Fixed Asset (container corner casting)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    # fixed_asset_init_pos_noise: list = [0.0, 0.0, 0.0]
    fixed_asset_init_orn_deg: float = 90.0  # Fixed asset yaw angle
    fixed_asset_init_orn_range_deg: float = 0.5  # Fixed asset yaw angle noise range
    # fixed_asset_init_orn_range_deg: float = 0.0  # Fixed asset yaw angle noise range

    # Held Asset (stacker)
    held_asset_pos_noise: list = [0.000, 0.000, 0.000]  # Noise level in gripper
    held_asset_rot_init: float = 0.0

    # Rewards - adjusted for stacker insertion
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.05  # Fraction of insertion depth
    engage_threshold: float = 0.5

    # Contact penalty for stacker insertion
    contact_penalty_scale: float = 0.15

    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Make it static
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.0937, 0.222, 1.09), 
            rot=(0.7071068, 0.0, 0.0 , 0.7071068)
        ),
    )
    
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.20, 1.21, 0.05), 
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={}
        ),
        actuators={},
    )
