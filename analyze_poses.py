#!/usr/bin/env python3

"""
Analyze the exact poses from the debug output to understand why gripping fails.
This script just does the math without running the simulation.
"""

import numpy as np

def analyze_poses():
    """Analyze the poses from the debug output."""
    
    print("=== POSE ANALYSIS FROM DEBUG OUTPUT ===\n")
    
    # Exact poses from your debug output
    fingertip_pos = np.array([0.5602, 0.7544, 0.7636])
    stacker_pos = np.array([0.5594, 0.7587, 0.7715])
    stacker_quat = np.array([0.7071, -0.7071, 0.0000, 0.0000])
    
    print(f"Fingertip position: {fingertip_pos}")
    print(f"Stacker position: {stacker_pos}")
    print(f"Stacker orientation (quat): {stacker_quat}")
    
    # Calculate distance
    distance = np.linalg.norm(fingertip_pos - stacker_pos)
    print(f"\nDistance between fingertip and stacker: {distance:.4f} meters")
    print(f"Distance in cm: {distance * 100:.2f} cm")
    
    # Check gripping feasibility
    gripping_threshold = 0.05  # 5cm
    if distance < gripping_threshold:
        print("✅ Objects are close enough for gripping")
    else:
        print(f"❌ Objects are too far apart for gripping (threshold: {gripping_threshold}m)")
    
    # Calculate relative position
    relative_pos = stacker_pos - fingertip_pos
    print(f"\nRelative position (stacker - fingertip): {relative_pos}")
    print(f"Relative position in cm: {relative_pos * 100}")
    
    # Analyze the relative position
    print(f"\n=== RELATIVE POSITION ANALYSIS ===")
    print(f"X offset: {relative_pos[0]:.4f}m ({relative_pos[0] * 100:.2f}cm)")
    print(f"Y offset: {relative_pos[1]:.4f}m ({relative_pos[1] * 100:.2f}cm)")
    print(f"Z offset: {relative_pos[2]:.4f}m ({relative_pos[2] * 100:.2f}cm)")
    
    # Check if stacker is above or below fingertip
    if relative_pos[2] > 0:
        print("❌ Stacker is ABOVE the fingertip (should be below for upside-down robot)")
    else:
        print("✅ Stacker is BELOW the fingertip (correct for upside-down robot)")
    
    # Check horizontal alignment
    horizontal_distance = np.sqrt(relative_pos[0]**2 + relative_pos[1]**2)
    print(f"Horizontal distance: {horizontal_distance:.4f}m ({horizontal_distance * 100:.2f}cm)")
    
    if horizontal_distance < 0.02:  # 2cm
        print("✅ Good horizontal alignment")
    else:
        print("❌ Poor horizontal alignment - stacker is too far horizontally")
    
    # Analyze the quaternion
    print(f"\n=== ORIENTATION ANALYSIS ===")
    print(f"Stacker quaternion: {stacker_quat}")
    
    # Convert quaternion to Euler angles for easier understanding
    # This is a simplified conversion for the specific quaternion
    w, x, y, z = stacker_quat
    
    # For quaternion [0.7071, -0.7071, 0.0, 0.0]
    # This represents a 90-degree rotation around the X-axis
    print("This quaternion represents a 90-degree rotation around the X-axis")
    
    # Check if this matches what you expect
    print("\n=== EXPECTED VS ACTUAL ===")
    print("Expected: Stacker should be below fingertip with same orientation as gripper")
    print("Actual: Stacker is above fingertip with 90-degree X rotation")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. The stacker is too far from the fingertip (8.1cm vs 5cm threshold)")
    print("2. The stacker is above the fingertip instead of below")
    print("3. The orientation might not match the gripper orientation")
    print("\nTo fix:")
    print("- Reduce the Z offset in get_handheld_asset_relative_pose()")
    print("- Make sure the relative position puts stacker below fingertip")
    print("- Consider using the same orientation as the gripper")

if __name__ == "__main__":
    analyze_poses()

