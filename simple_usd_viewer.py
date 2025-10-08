#!/usr/bin/env python3
"""
Simple USD asset viewer - load and show assets with CPU (no GPU).
"""

import time
from isaaclab.app import AppLauncher


# --- CPU-Konfiguration ---
CONFIG = {
    "headless": False,       # Fenster anzeigen
    "enable_gpu": False,     # keine GPU-Nutzung
    "device": "cpu",         # zwingt IsaacLab auf CPU
}
app_launcher = AppLauncher(CONFIG)
simulation_app = app_launcher.app

# Isaac/Omniverse Imports
import omni.usd
import omni.kit.commands
import omni.kit.viewport.utility as vp_utils
from pxr import Usd, UsdGeom, Gf, UsdLux

def get_asset_bounds(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] Prim at {prim_path} not found.")
        return None

    # Use Default time code instead of StageComputeBoundingBoxTime
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox.GetBox()


def main():
    print("Loading USD assets with CPU...")

    # Stage holen
    stage = omni.usd.get_context().get_stage()

    # Welt erstellen
    UsdGeom.Xform.Define(stage, "/World")

    # Assets laden
    omni.kit.commands.execute(
        "CreateReference",
        usd_context=omni.usd.get_context(),
        path_to="/World/Stacker",
        asset_path="/home/david/IsaacLab/source/isaaclab_assets/data/Factory/factory_stacker.usd",
    )

    # Add distant light (acts like a directional light)
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    xform = UsdGeom.Xformable(light)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 45, 0))


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

    # Print sizes

    stacker_bounds = get_asset_bounds(stage, "/World/Stacker")
    if stacker_bounds:
        size = stacker_bounds.GetMax() - stacker_bounds.GetMin()
        print(f"Size of /World/Stacker: {size}")

    corner_bounds = get_asset_bounds(stage, "/World/CornerCasting")
    if corner_bounds:
        size = corner_bounds.GetMax() - corner_bounds.GetMin()
        print(f"Size of /World/CornerCasting: {size}")

    print("Viewer running. Close the window to exit.")

    # --- Hier offen halten ---
    while simulation_app.is_running():
        simulation_app.update()

if __name__ == "__main__":
    main()
    simulation_app.close()
