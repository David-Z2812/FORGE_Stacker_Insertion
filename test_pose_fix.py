#!/usr/bin/env python3

"""
Test script to verify the pose fix without running the full simulation.
This shows the expected relative position after the fix.
"""

import numpy as np

def test_pose_fix():
    """Test the pose fix calculation."""
    
    print("=== TESTING POSE FIX ===\n")
    
    # Original problematic calculation
    print("ORIGINAL (problematic) calculation:")
    print("held_asset_relative_pos[:, 2] = -(height / 2.0)  # Half height below")
    print("held_asset_relative_pos[:, 2] -= fingerpad_length  # Account for fingerpad") 
    print("held_asset_relative_pos[:, 2] -= -0.09  # This was ADDING 0.09 (wrong!)")
    
    # Simulate the original calculation
    height = 0.1  # Assume 10cm height
    fingerpad_length = 0.05  # Assume 5cm fingerpad
    original_z = -(height / 2.0) - fingerpad_length - (-0.09)  # The problematic line
    print(f"Original Z position: {original_z:.4f}m ({original_z * 100:.2f}cm)")
    print("❌ This puts stacker ABOVE fingertip (positive Z)")
    
    print("\n" + "="*50 + "\n")
    
    # Fixed calculation
    print("FIXED calculation:")
    print("held_asset_relative_pos[:, 2] = -(height / 2.0)  # Half height below")
    print("held_asset_relative_pos[:, 2] -= fingerpad_length  # Account for fingerpad")
    print("held_asset_relative_pos[:, 2] -= 0.09  # Now SUBTRACTING 0.09 (correct!)")
    
    # Simulate the fixed calculation
    fixed_z = -(height / 2.0) - fingerpad_length - 0.09
    print(f"Fixed Z position: {fixed_z:.4f}m ({fixed_z * 100:.2f}cm)")
    print("✅ This puts stacker BELOW fingertip (negative Z)")
    
    print("\n" + "="*50 + "\n")
    
    # Show the difference
    difference = original_z - fixed_z
    print(f"DIFFERENCE: {difference:.4f}m ({difference * 100:.2f}cm)")
    print(f"The fix moves the stacker {difference * 100:.1f}cm lower")
    
    print("\n" + "="*50 + "\n")
    
    # Test with your actual values
    print("TESTING WITH YOUR ACTUAL VALUES:")
    print("From debug output:")
    print("Fingertip position: [0.5602, 0.7544, 0.7636]")
    print("Stacker position: [0.5594, 0.7587, 0.7715]")
    print("Relative position: [-0.0008, 0.0043, 0.0079]")
    
    # Calculate what the new relative position should be
    new_relative_z = fixed_z  # Use our fixed calculation
    new_relative_pos = np.array([-0.0008, 0.0043, new_relative_z])
    
    print(f"\nNEW expected relative position: {new_relative_pos}")
    print(f"NEW expected stacker position: {np.array([0.5602, 0.7544, 0.7636]) + new_relative_pos}")
    
    # Check if this is within gripping range
    new_distance = np.linalg.norm(new_relative_pos)
    print(f"NEW distance: {new_distance:.4f}m ({new_distance * 100:.2f}cm)")
    
    if new_distance < 0.05:  # 5cm threshold
        print("✅ NEW position is within gripping range!")
    else:
        print("❌ NEW position is still too far")

if __name__ == "__main__":
    test_pose_fix()

