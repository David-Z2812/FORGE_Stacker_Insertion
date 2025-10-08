#!/usr/bin/env python3
"""
Test script to check if the forge configuration can be imported correctly.
"""

def test_config_import():
    """Test importing the forge configuration."""
    
    print("🔍 Testing Forge Configuration Import...")
    
    try:
        # Test importing the forge tasks configuration
        print("📦 Importing forge_tasks_cfg...")
        from isaaclab_tasks.isaaclab_tasks.direct.forge.forge_tasks_cfg import (
            ForgeStackerInsert,
            Stacker,
            ContainerCornerCasting,
            LOCAL_ASSET_DIR
        )
        print("✓ forge_tasks_cfg imported successfully")
        
        # Test importing the forge environment configuration
        print("📦 Importing forge_env_cfg...")
        from isaaclab_tasks.isaaclab_tasks.direct.forge.forge_env_cfg import (
            ForgeTaskStackerInsertCfg
        )
        print("✓ forge_env_cfg imported successfully")
        
        # Test creating instances
        print("🏗️ Creating configuration instances...")
        stacker = Stacker()
        corner_casting = ContainerCornerCasting()
        stacker_insert = ForgeStackerInsert()
        env_cfg = ForgeTaskStackerInsertCfg()
        
        print("✓ All configuration instances created successfully")
        
        # Print configuration details
        print(f"\n📋 Configuration Details:")
        print(f"   Local Asset Directory: {LOCAL_ASSET_DIR}")
        print(f"   Stacker USD Path: {stacker.usd_path}")
        print(f"   Corner Casting USD Path: {corner_casting.usd_path}")
        print(f"   Task Name: {stacker_insert.name}")
        print(f"   Environment Task Name: {env_cfg.task_name}")
        print(f"   Episode Length: {env_cfg.episode_length_s}s")
        
        print("\n🎉 All configuration imports and instantiations successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config_import()






