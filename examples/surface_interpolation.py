#!/usr/bin/env python3
"""
Surface interpolation script for bathymetry data.
Creates interpolated surfaces using different methods and datasets:
- Full point cloud
- Subsampled point cloud
- Weighted subsampled point cloud (using point strengths)

Interpolation methods: median, shallowest, mean
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import argparse
from tqdm import tqdm
import time

# Add the parent directory to Python path so we can import bathy_subsample
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from Bathy_Subsample import IsolationGrid

class SurfaceInterpolator:
    """Handles surface interpolation with different methods."""
    
    def __init__(self, grid_resolution=1.0):
        """Initialize with grid resolution in meters."""
        self.grid_resolution = grid_resolution
    
    def create_grid(self, points):
        """Create regular grid for interpolation."""
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        # Add small buffer
        buffer = self.grid_resolution * 2
        x_min -= buffer
        x_max += buffer
        y_min -= buffer
        y_max += buffer
        
        # Create grid
        x_grid = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        y_grid = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        return X, Y, x_grid, y_grid
    
    def interpolate_weighted(self, points, weights, X, Y, method='linear'):
        """Optimized unified weighted interpolation function for all griddata methods."""
        print(f"Interpolating using weighted median in grid cells, then {method} interpolation...")
        print(f"  Input points: {len(points)}")
        print(f"  Weight range: {np.min(weights):.1f} - {np.max(weights):.1f}")
        print(f"  Unique weights: {len(np.unique(weights))}")
        
        Z = np.full(X.shape, np.nan)
        
        # Get grid cell size and grid bounds
        dx = self.grid_resolution
        dy = self.grid_resolution
        
        # Calculate grid bounds
        x_min = np.min(X) - dx/2
        y_min = np.min(Y) - dy/2
        
        # Pre-compute grid assignments for all points (much faster!)
        print("  Pre-computing grid assignments...")
        grid_i = np.floor((points[:, 1] - y_min) / dy).astype(int)
        grid_j = np.floor((points[:, 0] - x_min) / dx).astype(int)
        
        # Filter points that fall within the grid bounds
        valid_mask = ((grid_i >= 0) & (grid_i < X.shape[0]) & 
                     (grid_j >= 0) & (grid_j < X.shape[1]))
        
        if np.sum(valid_mask) == 0:
            print("  Warning: No points fall within grid bounds!")
            return Z
            
        valid_points = points[valid_mask]
        valid_weights = weights[valid_mask]
        valid_i = grid_i[valid_mask]
        valid_j = grid_j[valid_mask]
        
        print(f"  Processing {np.sum(valid_mask)} points within grid bounds...")
        
        # Group points by grid cell using unique combinations
        grid_coords = np.column_stack([valid_i, valid_j])
        unique_coords, inverse_indices = np.unique(grid_coords, axis=0, return_inverse=True)
        
        print(f"  Computing weighted medians for {len(unique_coords)} occupied grid cells...")
        
        # Process each unique grid cell
        for idx, (i, j) in enumerate(tqdm(unique_coords, desc="Processing grid cells")):
            # Get all points in this grid cell
            cell_mask = (inverse_indices == idx)
            cell_values = valid_points[cell_mask, 2]  # Z values
            cell_weights = valid_weights[cell_mask]
            
            # Calculate weighted median
            if len(cell_values) > 0:
                if np.sum(cell_weights) > 0 and np.max(cell_weights) > 0:
                    # Efficient weighted median using numpy operations
                    # Create weighted array by replicating based on integer weights
                    weighted_values = []
                    for value, weight in zip(cell_values, cell_weights):
                        replications = max(1, int(weight))
                        weighted_values.extend([value] * replications)
                    Z[i, j] = np.median(weighted_values)
                else:
                    Z[i, j] = np.median(cell_values)
        
        # Count filled cells
        filled_cells = np.sum(~np.isnan(Z))
        print(f"  Filled {filled_cells}/{Z.size} cells with weighted medians")
        
        # If there are empty cells, interpolate using filled cells
        if filled_cells < Z.size and filled_cells > 0:
            print(f"  Interpolating {Z.size - filled_cells} empty cells using {method}...")
            
            # Get coordinates and values of filled cells
            filled_mask = ~np.isnan(Z)
            filled_coords = np.column_stack([X[filled_mask], Y[filled_mask]])
            filled_values = Z[filled_mask]
            
            # Get coordinates of empty cells
            empty_mask = np.isnan(Z)
            empty_coords = np.column_stack([X[empty_mask], Y[empty_mask]])
            
            # Interpolate to empty cells
            if len(empty_coords) > 0:
                interpolated_values = griddata(filled_coords, filled_values, empty_coords, method=method)
                Z[empty_mask] = interpolated_values
        
        return Z
    
    def save_surface(self, X, Y, Z, filename):
        """Save surface as XYZ file."""
        print(f"Saving surface to {filename}...")
        
        # Flatten arrays and remove NaN values
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        valid_mask = ~np.isnan(z_flat)
        
        surface_data = np.column_stack([
            x_flat[valid_mask],
            y_flat[valid_mask], 
            z_flat[valid_mask]
        ])
        
        np.savetxt(filename, surface_data, fmt='%.6f', header='X Y Z')
        print(f"Saved {len(surface_data)} surface points")
    
    def create_surface_plot(self, X, Y, Z, title, filename, vmin=None, vmax=None):
        """Create and save surface plot with consistent color scale."""
        plt.figure(figsize=(12, 10))
        
        # Handle large coordinates by using offsets for cleaner display
        x_offset = np.round(np.min(X), -2)  # Round to nearest 100
        y_offset = np.round(np.min(Y), -2)  # Round to nearest 100
        
        X_plot = X - x_offset
        Y_plot = Y - y_offset
        
        # Create filled grid plot with consistent color scale
        mesh = plt.pcolormesh(X_plot, Y_plot, Z, cmap='viridis_r', shading='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, label='Depth (m)')
        
        plt.title(title)
        plt.xlabel(f'X (m) + {x_offset:,.0f}')
        plt.ylabel(f'Y (m) + {y_offset:,.0f}')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Force matplotlib to use fixed notation instead of scientific
        plt.ticklabel_format(style='plain', axis='both')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved surface plot: {filename}")
        if vmin is not None and vmax is not None:
            print(f"  Color range: {vmin:.2f} to {vmax:.2f} m")

    def create_difference_plot(self, X, Y, Z1, Z2, title, filename, label1="Dataset 1", label2="Dataset 2", vmin=None, vmax=None):
        """Create and save difference plot between two surfaces with consistent depth color scales."""
        plt.figure(figsize=(15, 5))
        
        # Handle large coordinates by using offsets for cleaner display
        x_offset = np.round(np.min(X), -2)
        y_offset = np.round(np.min(Y), -2)
        X_plot = X - x_offset
        Y_plot = Y - y_offset
        
        # Calculate difference (only where both surfaces have valid data)
        valid_mask = ~(np.isnan(Z1) | np.isnan(Z2))
        Z_diff = np.full(Z1.shape, np.nan)
        Z_diff[valid_mask] = Z1[valid_mask] - Z2[valid_mask]
        
        # Create three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: First surface with consistent color scale
        im1 = axes[0].pcolormesh(X_plot, Y_plot, Z1, cmap='viridis_r', shading='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title(label1)
        axes[0].set_xlabel(f'X (m) + {x_offset:,.0f}')
        axes[0].set_ylabel(f'Y (m) + {y_offset:,.0f}')
        axes[0].axis('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].ticklabel_format(style='plain', axis='both')
        plt.colorbar(im1, ax=axes[0], label='Depth (m)')
        
        # Plot 2: Second surface with consistent color scale
        im2 = axes[1].pcolormesh(X_plot, Y_plot, Z2, cmap='viridis_r', shading='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title(label2)
        axes[1].set_xlabel(f'X (m) + {x_offset:,.0f}')
        axes[1].set_ylabel(f'Y (m) + {y_offset:,.0f}')
        axes[1].axis('equal')
        axes[1].grid(True, alpha=0.3)
        axes[1].ticklabel_format(style='plain', axis='both')
        plt.colorbar(im2, ax=axes[1], label='Depth (m)')
        
        # Plot 3: Difference (keeps its own scale for better contrast)
        im3 = axes[2].pcolormesh(X_plot, Y_plot, Z_diff, cmap='RdBu_r', shading='auto')
        axes[2].set_title(f'Difference\n({label1} - {label2})')
        axes[2].set_xlabel(f'X (m) + {x_offset:,.0f}')
        axes[2].set_ylabel(f'Y (m) + {y_offset:,.0f}')
        axes[2].axis('equal')
        axes[2].grid(True, alpha=0.3)
        axes[2].ticklabel_format(style='plain', axis='both')
        cbar3 = plt.colorbar(im3, ax=axes[2], label='Difference (m)')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate difference statistics
        valid_diff = Z_diff[valid_mask]
        if len(valid_diff) > 0:
            stats_text = (f"Difference Statistics:\n"
                         f"  Mean: {np.mean(valid_diff):.3f} m\n"
                         f"  RMS: {np.sqrt(np.mean(valid_diff**2)):.3f} m\n"
                         f"  Std: {np.std(valid_diff):.3f} m\n"
                         f"  Min: {np.min(valid_diff):.3f} m\n"
                         f"  Max: {np.max(valid_diff):.3f} m\n"
                         f"  Valid cells: {len(valid_diff):,}/{Z1.size:,}")
            print(f"Saved difference plot: {filename}")
            if vmin is not None and vmax is not None:
                print(f"  Depth color range: {vmin:.2f} to {vmax:.2f} m")
            print(stats_text)
        else:
            print(f"Saved difference plot: {filename}")
            print("  Warning: No overlapping valid data for difference calculation")
        
        return Z_diff, valid_diff if len(valid_diff) > 0 else None

    def efficient_weighted_median(self, values, weights):
        """
        Calculate weighted median efficiently without replicating values.
        """
        if len(values) == 0:
            return np.nan
            
        # Convert weights to integers
        int_weights = np.maximum(1, weights.astype(int))
        
        # If all weights are 1, just return regular median
        if np.all(int_weights == 1):
            return np.median(values)
        
        # Sort values and corresponding weights
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = int_weights[sorted_indices]
        
        # Calculate cumulative weights
        cumsum_weights = np.cumsum(sorted_weights)
        total_weight = cumsum_weights[-1]
        
        # Find median position
        median_pos = total_weight / 2.0
        
        # Find the value at median position
        median_idx = np.searchsorted(cumsum_weights, median_pos, side='right')
        
        if median_idx < len(sorted_values):
            # If total weight is even and we're exactly at the boundary, average two middle values
            if total_weight % 2 == 0 and cumsum_weights[median_idx - 1] == median_pos:
                if median_idx < len(sorted_values):
                    return (sorted_values[median_idx - 1] + sorted_values[median_idx]) / 2
                else:
                    return sorted_values[median_idx - 1]
            else:
                return sorted_values[median_idx]
        else:
            return sorted_values[-1]

def load_points(filename):
    """Load point cloud from file."""
    print(f"Loading points from {filename}...")
    points = np.loadtxt(filename)
    print(f"Loaded {len(points)} points")
    return points

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create interpolated surfaces from point cloud data")
    parser.add_argument("--input", required=True, help="Input XYZ file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--grid_resolution", type=float, default=1.0, help="Grid resolution in meters (default: 2.0)")
    parser.add_argument("--group_size", type=int, default=1000, help="Group size for subsampling")
    parser.add_argument("--voxel_size", type=float, default=3.0, help="Voxel size for subsampling")
    parser.add_argument("--max_modes", type=int, default=2, help="Maximum modes for subsampling")
    parser.add_argument("--downsample_original", type=int, default=None, help="Downsample original dataset to this many points for faster processing")
    parser.add_argument("--navigation", type=bool, default=False, help="Navigation mode")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load original point cloud
    original_points = load_points(args.input)
    
    # Optionally downsample original points for faster processing
    if args.downsample_original and len(original_points) > args.downsample_original:
        print(f"Downsampling original dataset from {len(original_points):,} to {args.downsample_original:,} points for faster processing...")
        indices = np.random.choice(len(original_points), args.downsample_original, replace=False)
        original_points = original_points[indices]
        print(f"Original dataset downsampled to {len(original_points):,} points")
    
    # Process with isolation grid to get subsampled points
    print("\nProcessing with IsolationGrid...")
    processor = IsolationGrid(
        group_size=args.group_size,
        voxel_x_size=args.voxel_size,
        voxel_y_size=args.voxel_size,
        mode_probability_threshold=1.0,
        anomaly_threshold=0.9,
        min_points_for_mode=5,
        max_modes=args.max_modes,
        best_hypothesis=True,
        verbose=False,
        save_intermediate_files=False,
        navigation=args.navigation
    )
    
    # IsolationGrid.process() expects a filename, so we need to use the input file directly
    subsampled_points, stats = processor.process(args.input, args.output)
    point_strengths = stats.get('point_strengths', np.ones(len(subsampled_points)))
    
    print(f"Subsampled: {len(original_points)} -> {len(subsampled_points)} points")
    print(f"Reduction: {stats['reduction_percentage']:.1f}%")
    
    # Calculate color range from original point cloud for consistent scaling
    original_depths = original_points[:, 2]
    depth_vmin = np.min(original_depths)
    depth_vmax = np.max(original_depths) 
    print(f"Original point cloud depth range: {depth_vmin:.2f} to {depth_vmax:.2f} m")
    print(f"This range will be used for all color scales")
    
    # Save subsampled points for reference
    np.savetxt(f"{args.output}/subsampled_points.xyz", subsampled_points, fmt='%.6f')
    
    # Initialize interpolator
    interpolator = SurfaceInterpolator(grid_resolution=args.grid_resolution)
    
    # Create grid based on original points extent
    print(f"\nCreating interpolation grid (resolution: {args.grid_resolution}m)...")
    X, Y, x_grid, y_grid = interpolator.create_grid(original_points)
    print(f"Grid size: {X.shape[1]} x {X.shape[0]} = {X.size:,} cells")
    print(f"Grid cell size: {args.grid_resolution}m x {args.grid_resolution}m")
    
    # Define datasets with appropriate weights
    # First, create the strength > 1 dataset
    strength_gt_1_mask = point_strengths > 1
    strength_gt_1_points = subsampled_points[strength_gt_1_mask]
    strength_gt_1_strengths = point_strengths[strength_gt_1_mask]
    
    print(f"Points with strength > 1: {len(strength_gt_1_points)} out of {len(subsampled_points)} subsampled points")
    
    datasets = {
        'original': (original_points, np.ones(len(original_points)), "Original Point Cloud"),
        'subsampled': (subsampled_points, np.ones(len(subsampled_points)), "Subsampled Point Cloud"),
        'weighted': (subsampled_points, point_strengths, "Weighted Subsampled Point Cloud")
    }
    
    # Only add strength > 1 dataset if we have points
    if len(strength_gt_1_points) > 0:
        datasets['strength_gt_1'] = (strength_gt_1_points, strength_gt_1_strengths, "Points with Strength > 1")
    else:
        print("Warning: No points with strength > 1 found, skipping this dataset")
    
    # Define interpolation methods
    methods = ['linear', 'cubic', 'nearest']
    
    # Process each combination
    print("\n" + "="*60)
    print("STARTING SURFACE INTERPOLATION")
    print("="*60)
    
    results = {}
    
    for dataset_name, (points, weights, description) in datasets.items():
        print(f"\n{'-'*40}")
        print(f"Processing: {description}")
        print(f"Points: {len(points):,}")
        print(f"{'-'*40}")
        
        results[dataset_name] = {}
        
        for method in methods:
            print(f"\nMethod: {method.upper()}")
            start_time = time.time()
            
            # Use unified weighted interpolation for all cases
            Z = interpolator.interpolate_weighted(points, weights, X, Y, method=method)
            
            end_time = time.time()
            
            # Store result
            results[dataset_name][method] = Z
            
            # Calculate coverage
            valid_cells = np.sum(~np.isnan(Z))
            coverage = (valid_cells / Z.size) * 100
            
            print(f"Completed in {end_time - start_time:.1f} seconds")
            print(f"Coverage: {valid_cells:,}/{Z.size:,} cells ({coverage:.1f}%)")
            
            # Save surface
            surface_filename = f"{args.output}/surface_{dataset_name}_{method}.xyz"
            interpolator.save_surface(X, Y, Z, surface_filename)
            
            # Create plot
            plot_title = f"{description}\n{method.title()} Interpolation"
            plot_filename = f"{args.output}/surface_{dataset_name}_{method}.png"
            interpolator.create_surface_plot(X, Y, Z, plot_title, plot_filename, vmin=depth_vmin, vmax=depth_vmax)
    
    # Create comparison plots
    print(f"\n{'-'*40}")
    print("Creating comparison plots...")
    print(f"{'-'*40}")
    
    # Calculate coordinate offsets for consistent plotting
    x_offset = np.round(np.min(X), -2)  # Round to nearest 100
    y_offset = np.round(np.min(Y), -2)  # Round to nearest 100
    X_plot = X - x_offset
    Y_plot = Y - y_offset
    
    for method in methods:
        num_datasets = len(datasets)
        fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 6))
        
        # Handle single subplot case
        if num_datasets == 1:
            axes = [axes]
        
        for i, (dataset_name, (_, _, description)) in enumerate(datasets.items()):
            if method in results[dataset_name]:
                Z = results[dataset_name][method]
                title = f"{description}\n{method.title()}"
            else:
                # Skip if method not available for this dataset
                Z = np.full(X.shape, np.nan)
                title = f"{description}\n{method.title()} (Not Available)"
            
            im = axes[i].pcolormesh(X_plot, Y_plot, Z, cmap='viridis_r', shading='auto', vmin=depth_vmin, vmax=depth_vmax)
            axes[i].set_title(title)
            axes[i].set_xlabel(f'X (m) + {x_offset:,.0f}')
            axes[i].set_ylabel(f'Y (m) + {y_offset:,.0f}')
            axes[i].axis('equal')
            axes[i].grid(True, alpha=0.3)
            axes[i].ticklabel_format(style='plain', axis='both')
        
        # Add shared colorbar
        plt.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Depth (m)')
        
        plt.suptitle(f'Surface Comparison - {method.title()} Interpolation', fontsize=16)
        
        comparison_filename = f"{args.output}/comparison_{method}.png"
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {comparison_filename}")
        print(f"  Coordinate offsets applied: X+{x_offset:,.0f}, Y+{y_offset:,.0f}")
    
    # Create difference plots
    print(f"\n{'-'*40}")
    print("Creating difference plots...")
    print(f"{'-'*40}")
    
    # Create difference plots for each method
    for method in ['linear', 'cubic', 'nearest']:
        if method in results['original'] and method in results['subsampled']:
            Z1 = results['original'][method]
            Z2 = results['subsampled'][method]
            title = f"{method.title()} - Original vs Subsampled"
            filename = f"{args.output}/difference_{method}_original_vs_subsampled.png"
            interpolator.create_difference_plot(
                X, Y, Z1, Z2, title, filename,
                label1="Original", label2="Subsampled", vmin=depth_vmin, vmax=depth_vmax
            )
        
        if method in results['original'] and method in results['weighted']:
            Z1 = results['original'][method]
            Z2 = results['weighted'][method]
            title = f"{method.title()} - Original vs Weighted"
            filename = f"{args.output}/difference_{method}_original_vs_weighted.png"
            interpolator.create_difference_plot(
                X, Y, Z1, Z2, title, filename,
                label1="Original", label2="Weighted", vmin=depth_vmin, vmax=depth_vmax
            )
        
        if method in results['subsampled'] and method in results['weighted']:
            Z1 = results['subsampled'][method]
            Z2 = results['weighted'][method]
            title = f"{method.title()} - Subsampled vs Weighted"
            filename = f"{args.output}/difference_{method}_subsampled_vs_weighted.png"
            interpolator.create_difference_plot(
                X, Y, Z1, Z2, title, filename,
                label1="Subsampled", label2="Weighted", vmin=depth_vmin, vmax=depth_vmax
            )

        if method in results['original'] and method in results['strength_gt_1']:
            Z1 = results['original'][method]
            Z2 = results['strength_gt_1'][method]
            title = f"{method.title()} - Original vs Strength > 1"
            filename = f"{args.output}/difference_{method}_original_vs_strength_gt_1.png"
            interpolator.create_difference_plot(
                X, Y, Z1, Z2, title, filename,
                label1="Original", label2="Strength > 1", vmin=depth_vmin, vmax=depth_vmax
            )
        
    
    # Compare griddata methods
    if 'linear' in results['original'] and 'cubic' in results['original']:
        Z1 = results['original']['linear']
        Z2 = results['original']['cubic']
        title = "Original Data - Linear vs Cubic"
        filename = f"{args.output}/difference_linear_vs_cubic.png"
        interpolator.create_difference_plot(
            X, Y, Z1, Z2, title, filename,
            label1="Linear", label2="Cubic", vmin=depth_vmin, vmax=depth_vmax
        )
    
    # Compare weighted linear vs cubic
    if 'linear' in results['weighted'] and 'cubic' in results['weighted']:
        Z1 = results['weighted']['linear']
        Z2 = results['weighted']['cubic']
        title = "Weighted Data - Linear vs Cubic"
        filename = f"{args.output}/difference_weighted_linear_vs_cubic.png"
        interpolator.create_difference_plot(
            X, Y, Z1, Z2, title, filename,
            label1="Linear", label2="Cubic", vmin=depth_vmin, vmax=depth_vmax
        )
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    for dataset_name, (points, weights, description) in datasets.items():
        print(f"\n{description}:")
        print(f"  Points: {len(points):,}")
        
        if dataset_name == 'weighted' and weights is not None:
            print(f"  Weight range: {np.min(weights):.1f} - {np.max(weights):.1f}")
            print(f"  Mean weight: {np.mean(weights):.1f}")
        
        for method in methods:
            if method in results[dataset_name]:
                Z = results[dataset_name][method]
                valid_cells = np.sum(~np.isnan(Z))
                coverage = (valid_cells / Z.size) * 100
                
                valid_depths = Z[~np.isnan(Z)]
                if len(valid_depths) > 0:
                    print(f"  {method.title()}:")
                    print(f"    Coverage: {coverage:.1f}%")
                    print(f"    Depth range: {np.min(valid_depths):.2f} to {np.max(valid_depths):.2f} m")
                    print(f"    Mean depth: {np.mean(valid_depths):.2f} m")
    
    print(f"\nAll outputs saved to: {args.output}")
    print("Surface files: surface_[dataset]_[method].xyz")
    print("Plot files: surface_[dataset]_[method].png")
    print("Comparison files: comparison_[method].png")
    print("Difference files: difference_[type]_[datasets/methods].png")
    print("\nInterpolation Methods:")
    print("  Linear: Weighted median within cells, linear interpolation to fill gaps")
    print("  Cubic: Weighted median within cells, cubic interpolation to fill gaps")
    print("  Nearest: Weighted median within cells, nearest neighbor to fill gaps")
    print("\nDatasets:")
    print("  Original: All points with equal weight (1.0)")
    print("  Subsampled: Processed points with equal weight (1.0)")
    print("  Weighted: Processed points with actual strength values")
    if 'strength_gt_1' in datasets:
        print("  Strength > 1: Points with strength > 1, using their actual strength values")
    print("\nDifference Plots Created:")
    print("  Dataset comparisons: original vs subsampled/weighted")
    print("  Method comparisons: linear vs cubic")
    print("\nWeighted Methods:")
    print("  Uses point strengths from bathymetry processing:")
    print("  - Anomaly points (score > 0.5): strength = 1")
    print("  - Mode representatives: strength = number of points in mode")
    print("  All methods use weighted median within grid cells by replicating points by strength")

if __name__ == "__main__":
    main() 