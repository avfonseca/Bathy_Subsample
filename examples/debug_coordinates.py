#!/usr/bin/env python3
"""
Debug script to check coordinate system issues.
"""

import numpy as np

# Load the data
print("Loading example_points.xyz...")
points = np.loadtxt("example_points.xyz")

print(f"\nData shape: {points.shape}")
print(f"Number of points: {len(points)}")

print(f"\nX coordinates:")
print(f"  Min: {np.min(points[:, 0]):.2f}")
print(f"  Max: {np.max(points[:, 0]):.2f}")
print(f"  Range: {np.max(points[:, 0]) - np.min(points[:, 0]):.2f}")

print(f"\nY coordinates:")
print(f"  Min: {np.min(points[:, 1]):.2f}")
print(f"  Max: {np.max(points[:, 1]):.2f}")
print(f"  Range: {np.max(points[:, 1]) - np.min(points[:, 1]):.2f}")

print(f"\nZ coordinates:")
print(f"  Min: {np.min(points[:, 2]):.2f}")
print(f"  Max: {np.max(points[:, 2]):.2f}")
print(f"  Range: {np.max(points[:, 2]) - np.min(points[:, 2]):.2f}")

# Show first few points
print(f"\nFirst 5 points:")
for i in range(min(5, len(points))):
    print(f"  {points[i, 0]:.2f}, {points[i, 1]:.2f}, {points[i, 2]:.2f}")

# Test grid creation
print(f"\nTesting grid creation with 1m resolution...")
x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

grid_resolution = 1.0
buffer = grid_resolution * 2

x_min_grid = x_min - buffer
x_max_grid = x_max + buffer
y_min_grid = y_min - buffer  
y_max_grid = y_max + buffer

print(f"Grid extent:")
print(f"  X: {x_min_grid:.2f} to {x_max_grid:.2f}")
print(f"  Y: {y_min_grid:.2f} to {y_max_grid:.2f}")

x_grid = np.arange(x_min_grid, x_max_grid + grid_resolution, grid_resolution)
y_grid = np.arange(y_min_grid, y_max_grid + grid_resolution, grid_resolution)

print(f"Grid size: {len(x_grid)} x {len(y_grid)} = {len(x_grid) * len(y_grid):,} cells")
print(f"X grid range: {x_grid[0]:.2f} to {x_grid[-1]:.2f}")
print(f"Y grid range: {y_grid[0]:.2f} to {y_grid[-1]:.2f}") 