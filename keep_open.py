#!/usr/bin/env python3
"""
USD asset viewer that keeps the window open.
"""

import os
import time

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, Gf, UsdLux

def main():
    print("Loading USD assets...")
    
    # Get stage
    stage = omni.usd.get_context().get_stage()
    
    # Create world
    world = UsdGeom.Xform.Define(stage, "/World")
    
    # Add light
    light = UsdLux.DirectionalLight.Define(stage, "/World/Light")
    light.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))
    
    # Add camera
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3f(0.3, -0.5, 0.2))
    
    # Load stacker
    omni.kit.commands.execute(
        "CreateReference",
        usd_context=omni.usd.get_context(),
        path_to="/World/Stacker",
        asset_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/factory_stacker.usd"
    )
    
    # Load corner casting
    omni.kit.commands.execute(
        "CreateReference", 
        usd_context=omni.usd.get_context(),
        path_to="/World/CornerCasting",
        asset_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/factory_container_corner_casting.usd"
    )
    
    # Set camera
    omni.kit.commands.execute("SetActiveCamera", camera_path="/World/Camera")
    
    print("Assets loaded! You should see them in Isaac Sim.")
    print("Press Ctrl+C to exit.")
    
    # Keep the app running indefinitely
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        simulation_app.close()






