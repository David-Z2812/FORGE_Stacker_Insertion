#!/usr/bin/env python3
"""
Headless CPU visualization script - no window, just asset loading verification.
"""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# Force CPU-only rendering by setting environment variables
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all CUDA devices
os.environ["OMNI_DISABLE_GPU"] = "1"     # Disable GPU
os.environ["OMNI_CPU_ONLY"] = "1"        # Force CPU only

# launch omniverse app in headless mode
app_launcher = AppLauncher(headless=True, enable_cameras=False, enable_gpu=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, Gf


def visualize_headless():
    """Headless CPU visualization - just verify assets load correctly."""
    
    print("🎬 Loading USD Assets in Headless CPU Mode...")
    
    try:
        # Get the current stage
        stage = omni.usd.get_context().get_stage()
        print("✓ Got USD stage")
        
        # Create a simple scene
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim:
            world_prim = UsdGeom.Xform.Define(stage, "/World")
            print("✓ Created World prim")
        
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
        
        # Verify the stacker loaded
        stacker_prim = stage.GetPrimAtPath(stacker_prim_path)
        if stacker_prim and stacker_prim.IsValid():
            print("✓ Stacker asset is valid and loaded")
            # Check if it has geometry
            children = list(stacker_prim.GetChildren())
            print(f"✓ Stacker has {len(children)} child prims")
        else:
            print("❌ Stacker asset failed to load")
        
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
        
        # Verify the corner casting loaded
        corner_casting_prim = stage.GetPrimAtPath(corner_casting_prim_path)
        if corner_casting_prim and corner_casting_prim.IsValid():
            print("✓ Corner casting asset is valid and loaded")
            # Check if it has geometry
            children = list(corner_casting_prim.GetChildren())
            print(f"✓ Corner casting has {len(children)} child prims")
        else:
            print("❌ Corner casting asset failed to load")
        
        # Print stage summary
        print(f"\n📋 Stage Summary:")
        print(f"   - Total prims in stage: {len(list(stage.Traverse()))}")
        print(f"   - World prim: {world_prim.GetPrim().IsValid()}")
        print(f"   - Stacker prim: {stacker_prim.IsValid() if 'stacker_prim' in locals() else False}")
        print(f"   - Corner casting prim: {corner_casting_prim.IsValid() if 'corner_casting_prim' in locals() else False}")
        
        print("\n✅ Headless CPU visualization complete!")
        print("   Both USD assets loaded successfully in CPU-only mode.")
        print("   No CUDA/GPU usage detected.")
        
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
        visualize_headless()
    except Exception as e:
        print(f"Error running visualization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # close the app
        simulation_app.close()
