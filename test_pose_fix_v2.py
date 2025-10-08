#!/usr/bin/env python3

"""
Test script to verify the updated pose fix.
"""

import numpy as np

def test_pose_fix_v2():
    """Test the updated pose fix calculation."""
    
    print("=== TESTING UPDATED POSE FIX ===\n")
    
    # Updated calculation
    print("UPDATED calculation:")
    print("held_asset_relative_pos[:, 2] = -(height / 2.0)  # Half height below")
    print("held_asset_relative_pos[:, 2] -= fingerpad_length  # Account for fingerpad")
    print("held_asset_relative_pos[:, 2] -= 0.01  # Small gap (1cm below)")
    
    # Simulate the updated calculation
    height = 0.1  # Assume 10cm height
    fingerpad_length = 0.05  # Assume 5cm fingerpad
    updated_z = -(height / 2.0) - fingerpad_length - 0.01
    print(f"Updated Z position: {updated_z:.4f}m ({updated_z * 100:.2f}cm)")
    print("✅ This puts stacker BELOW fingertip (negative Z)")
    
    print("\n" + "="*50 + "\n")
    
    # Test with your actual values
    print("TESTING WITH YOUR ACTUAL VALUES:")
    print("From debug output:")
    print("Fingertip position: [0.5602, 0.7544, 0.7636]")
    
    # Calculate what the new relative position should be
    new_relative_z = updated_z  # Use our updated calculation
    new_relative_pos = np.array([-0.0008, 0.0043, new_relative_z])
    
    print(f"\nNEW expected relative position: {new_relative_pos}")
    new_stacker_pos = np.array([0.5602, 0.7544, 0.7636]) + new_relative_pos
    print(f"NEW expected stacker position: {new_stacker_pos}")
    
    # Check if this is within gripping range
    new_distance = np.linalg.norm(new_relative_pos)
    print(f"NEW distance: {new_distance:.4f}m ({new_distance * 100:.2f}cm)")
    
    if new_distance < 0.05:  # 5cm threshold
        print("✅ NEW position is within gripping range!")
    else:
        print("❌ NEW position is still too far")
    
    # Show the improvement
    original_distance = 0.0090  # From your debug output
    improvement = original_distance - new_distance
    print(f"\nIMPROVEMENT:")
    print(f"Original distance: {original_distance:.4f}m ({original_distance * 100:.2f}cm)")
    print(f"New distance: {new_distance:.4f}m ({new_distance * 100:.2f}cm)")
    print(f"Improvement: {improvement:.4f}m ({improvement * 100:.2f}cm)")

if __name__ == "__main__":
    test_pose_fix_v2()

