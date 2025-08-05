#!/usr/bin/env python3
"""
Helper script to run surface interpolation with comprehensive analysis.
Creates interpolated surfaces using both grid cell and griddata methods,
plus difference plots for quality comparison.
"""

import subprocess
import sys
import os

def main():
    # Configuration
    input_file = "norbit.txt"  # Replace with your file
    output_dir = "surface_analysis_output"
    
    # Parameters
    grid_resolution = 1.0  # meters
    group_size = 1000
    voxel_size = 3.0
    max_modes = 3
    downsample_original = 10000  # Downsample original to 50k points for faster processing
    navigation = True

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please update the input_file variable in this script.")
        return 1
    
    # Build command
    cmd = [
        sys.executable, "surface_interpolation.py",
        "--input", input_file,
        "--output", output_dir,
        "--grid_resolution", str(grid_resolution),
        "--group_size", str(group_size),
        "--voxel_size", str(voxel_size),
        "--max_modes", str(max_modes),
        "--downsample_original", str(downsample_original),
        "--navigation", str(navigation)
    ]
    
    print("Running comprehensive surface interpolation analysis...")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Grid resolution: {grid_resolution}m")
    #print(f"Original dataset will be downsampled to {downsample_original:,} points for faster processing")
    print("\nThis will create:")
    print("• GridData interpolations (linear, cubic, nearest, shallowest)")
    print("• Weighted interpolations using point strengths (mean & median)")
    print("• Individual surface plots for each method/dataset")
    print("• Comparison plots showing all datasets side-by-side")
    print("• Difference plots showing quantitative differences")
    print("• Surface files in XYZ format")
    print()
    
    # Run the analysis
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nAll outputs saved to: {output_dir}/")
        print("\nFile types created:")
        print("  surface_*.xyz - Interpolated surface data")
        print("  surface_*.png - Individual surface plots")
        print("  comparison_*.png - Side-by-side dataset comparisons")
        print("  difference_*.png - Quantitative difference analysis")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running analysis: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 