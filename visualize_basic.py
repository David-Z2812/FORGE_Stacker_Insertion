#!/usr/bin/env python3
"""
Basic visualization script that loads the USD assets directly without Isaac Lab environment.
This bypasses the articulation requirements and shows the assets in Isaac Sim.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# launch omniverse app with rendering enabled
app_launcher = AppLauncher(headless=False)  # Set to False for visualization
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, Gf


def visualize_basic():
    """Basic visualization of the USD assets in Isaac Sim."""
    
    print("🎬 Loading USD Assets for Basic Visualization...")
    
    try:
        # Get the current stage
        stage = omni.usd.get_context().get_stage()
        print("✓ Got USD stage")
        
        # Create a simple scene
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim:
            world_prim = UsdGeom.Xform.Define(stage, "/World")
            print("✓ Created World prim")
        
        # Add a ground plane
        ground_prim = UsdGeom.Xform.Define(stage, "/World/Ground")
        ground_prim.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0))
        print("✓ Created ground plane")
        
        # Get USD context
        usd_context = omni.usd.get_context()
        
        # Load the stacker asset
        stacker_path = "/home/david/IsaacLab/source/isaaclab_assets/data/Factory/factory_stacker.usd"
        stacker_prim_path = "/World/Stacker"
        
        # Use USD commands to load the asset
        omni.kit.commands.execute(
            "CreateReference",
            usd_context=usd_context,
            path_to=stacker_prim_path,
            asset_path=stacker_path,
            instanceable=False
        )
        print(f"✓ Loaded stacker asset from {stacker_path}")
        
        # Position the stacker
        stacker_prim = stage.GetPrimAtPath(stacker_prim_path)
        if stacker_prim:
            xform = UsdGeom.Xformable(stacker_prim)
            # Get existing translate op or create one
            translate_op = xform.GetTranslateOp()
            if translate_op:
                translate_op.Set(Gf.Vec3f(0.0, 0.0, 0.05))  # 5cm above ground
            else:
                xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.05))
            print("✓ Positioned stacker")
        
        # Load the container corner casting asset
        corner_casting_path = "/home/david/IsaacLab/source/isaaclab_assets/data/Factory/factory_container_corner_casting.usd"
        corner_casting_prim_path = "/World/CornerCasting"
        
        # Use USD commands to load the asset
        omni.kit.commands.execute(
            "CreateReference",
            usd_context=usd_context,
            path_to=corner_casting_prim_path,
            asset_path=corner_casting_path,
            instanceable=False
        )
        print(f"✓ Loaded corner casting asset from {corner_casting_path}")
        
        # Position the corner casting
        corner_casting_prim = stage.GetPrimAtPath(corner_casting_prim_path)
        if corner_casting_prim:
            xform = UsdGeom.Xformable(corner_casting_prim)
            # Get existing translate op or create one
            translate_op = xform.GetTranslateOp()
            if translate_op:
                translate_op.Set(Gf.Vec3f(0.6, 0.0, 0.05))  # 60cm away, 5cm above ground
            else:
                xform.AddTranslateOp().Set(Gf.Vec3f(0.6, 0.0, 0.05))
            print("✓ Positioned corner casting")
        
        print("\n🎮 Isaac Sim Visualization:")
        print("   - You should see both USD assets in Isaac Sim")
        print("   - Stacker at position (0, 0, 0.05)")
        print("   - Corner casting at position (0.6, 0, 0.05)")
        print("   - You can rotate, zoom, and pan the view")
        print("   - Press Ctrl+C to exit")
        
        # Keep the simulation running indefinitely
        import time
        print("\n⏱️  Keeping simulation running indefinitely...")
        print("   Press Ctrl+C to exit when you're done inspecting the assets")
        try:
            while True:
                time.sleep(1)  # Sleep for 1 second at a time
        except KeyboardInterrupt:
            print("\n⏹️  Stopping visualization...")
        
        print("\n✅ Basic visualization complete!")
        print("   You should have seen both USD assets in Isaac Sim.")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # run the visualization
        visualize_basic()
    except Exception as e:
        print(f"Error running visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the app
        simulation_app.close()
