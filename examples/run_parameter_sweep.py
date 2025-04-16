#!/usr/bin/env python3
"""
Example script to run the parameter sweep for IsolationGrid.
This script demonstrates how to use the parameter_sweep.py script with a specific example.
"""

import os
import sys
import subprocess

# Add the parent directory to Python path so we can import bathy_subsample
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

def main():
    """Run the parameter sweep with example parameters."""
    # Get the path to the example points file
    input_file = os.path.join(current_dir, "example_points.xyz")
    
    # Create output directory
    output_dir = os.path.join(current_dir, "parameter_sweep_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define default parameters
    default_params = {
        "group_size": 1000,
        "voxel_size": 1.0,
        "anomaly_threshold": 0.5,
        "mode_probability_threshold": 0.3,
        "min_points_for_mode": 3,
        "max_modes": 2
    }
    
    # Define parameter ranges to sweep (one at a time)
    param_ranges = {
        "group_size": "1000,2000,5000,10000,20000",
        "voxel_size": "0.5,1.0,2.0,3.0,5.0",
        "anomaly_threshold": "0.2,0.3,0.4,0.5,0.6,0.7",
        "mode_probability_threshold": "1,2,3,4,5",
        "min_points_for_mode": "2,3,5,10",
        "max_modes": "1,2,3,4,5"
    }
    
    # Run parameter sweep for each parameter
    for param_name, param_values in param_ranges.items():
        print(f"\nSweeping parameter: {param_name}")
        
        # Create output directory for this parameter
        param_output_dir = os.path.join(output_dir, param_name)
        os.makedirs(param_output_dir, exist_ok=True)
        
        # Build the command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            os.path.join(current_dir, "parameter_sweep.py"),
            "--input", input_file,
            "--output", param_output_dir
        ]
        
        # Add the parameter to sweep
        if param_name == "voxel_size":
            # Set both voxel_x_size and voxel_y_size to the same value
            cmd.extend(["--voxel_x_size", param_values])
            cmd.extend(["--voxel_y_size", param_values])
        else:
            cmd.extend([f"--{param_name}", param_values])
        
        # Run the command
        print(f"Running parameter sweep for {param_name} with command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        print(f"Parameter sweep for {param_name} completed. Results saved to {param_output_dir}")
    
    print(f"\nAll parameter sweeps completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 