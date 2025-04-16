#!/usr/bin/env python3
"""
Script to perform a parameter sweep for the IsolationGrid class.
This script runs the IsolationGrid with different parameter combinations and saves the results.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import itertools
from datetime import datetime
import contextlib
import io
import time

# Add the parent directory to Python path so we can import bathy_subsample
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from Bathy_Subsample import IsolationGrid

# Utility functions for loading and saving XYZ files
def load_xyz_file(filename):
    """Load points from an XYZ file."""
    return np.loadtxt(filename)

def save_xyz_file(points, filename):
    """Save points to an XYZ file."""
    np.savetxt(filename, points)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parameter sweep for IsolationGrid")
    parser.add_argument("--input", required=True, help="Input XYZ file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--group_size", help="Comma-separated list of group sizes")
    parser.add_argument("--voxel_x_size", help="Comma-separated list of voxel x sizes")
    parser.add_argument("--voxel_y_size", help="Comma-separated list of voxel y sizes")
    parser.add_argument("--anomaly_threshold", help="Comma-separated list of anomaly thresholds")
    parser.add_argument("--mode_probability_threshold", help="Comma-separated list of mode probability thresholds")
    parser.add_argument("--min_points_for_mode", help="Comma-separated list of minimum points for mode")
    parser.add_argument("--max_modes", help="Comma-separated list of maximum modes")
    return parser.parse_args()

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to temporarily suppress stdout."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def run_isolation_grid(points, params):
    """Run IsolationGrid with the given parameters."""
    with suppress_stdout():
        grid = IsolationGrid(**params)
        # The IsolationGrid class doesn't have a fit method, it uses process directly
        # Create a temporary output directory for processing
        temp_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_output")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Save points to a temporary file
        temp_input_file = os.path.join(temp_output_dir, "temp_input.xyz")
        save_xyz_file(points, temp_input_file)
        
        # Process the points
        final_points, stats = grid.process(temp_input_file, output_dir=temp_output_dir)
        
        # Clean up temporary files and directory
        if os.path.exists(temp_output_dir) and os.path.isdir(temp_output_dir):
            # Remove all files in the directory
            for filename in os.listdir(temp_output_dir):
                file_path = os.path.join(temp_output_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            
            # Remove the directory itself
            try:
                os.rmdir(temp_output_dir)
            except Exception as e:
                print(f'Failed to delete directory {temp_output_dir}. Reason: {e}')
                
        return final_points, stats

def plot_results(results, output_dir, param_name):
    """Plot results for a single parameter sweep."""
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract parameter values and metrics
    param_values = []
    runtimes = []
    high_anomaly_ratios = []
    std_unimodal_multimodal_ratios = []
    voxel_to_output_ratios = []
    reduction_ratios = []
    outside_points_ratios = []
    
    for result in results:
        param_values.append(result["param_value"])
        runtimes.append(result["runtime"])
        high_anomaly_ratios.append(result["high_anomaly_ratio"])
        std_unimodal_multimodal_ratios.append(result["std_unimodal_multimodal_ratio"])
        voxel_to_output_ratios.append(result["voxel_to_output_ratio"])
        reduction_ratios.append(result["reduction_ratio"])
        outside_points_ratios.append(result["outside_points_ratio"])
    
    # Plot 1: Parameter vs runtime
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, runtimes, 'o-')
    plt.xlabel(param_name)
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Parameter vs Runtime")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"runtime_vs_{param_name}.png"))
    plt.close()
    
    # Plot 2: Parameter vs ratio High Anomaly/(Std + unimodal + multimodal)
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, high_anomaly_ratios, 'o-')
    plt.xlabel(param_name)
    plt.ylabel("High Anomaly / (Std + Unimodal + Multimodal)")
    plt.title(f"Parameter vs High Anomaly Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"high_anomaly_ratio_vs_{param_name}.png"))
    plt.close()
    
    # Plot 3: Parameter vs ratio (Std + unimodal)/multimodal
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, std_unimodal_multimodal_ratios, 'o-')
    plt.xlabel(param_name)
    plt.ylabel("(Std + Unimodal) / Multimodal")
    plt.title(f"Parameter vs (Std + Unimodal)/Multimodal Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"std_unimodal_multimodal_ratio_vs_{param_name}.png"))
    plt.close()
    
    # Plot 4: Parameter vs (Std+Unimodal+Multimodal)/total output points
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, voxel_to_output_ratios, 'o-')
    plt.xlabel(param_name)
    plt.ylabel("(Std + Unimodal + Multimodal) / Total Output Points")
    plt.title(f"Parameter vs Voxel to Output Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"voxel_to_output_ratio_vs_{param_name}.png"))
    plt.close()
    
    # Plot 5: Parameter vs reduction
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, reduction_ratios, 'o-')
    plt.xlabel(param_name)
    plt.ylabel("Reduction Ratio")
    plt.title(f"Parameter vs Reduction")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"reduction_vs_{param_name}.png"))
    plt.close()
    
    # Plot 6: Parameter vs outside points/(std+unimodal+multimodal)
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, outside_points_ratios, 'o-')
    plt.xlabel(param_name)
    plt.ylabel("Outside Points / (Std + Unimodal + Multimodal)")
    plt.title(f"Parameter vs Outside Points Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"outside_points_ratio_vs_{param_name}.png"))
    plt.close()
    
    # Save results to CSV
    df = pd.DataFrame({
        param_name: param_values,
        "runtime": runtimes,
        "high_anomaly_ratio": high_anomaly_ratios,
        "std_unimodal_multimodal_ratio": std_unimodal_multimodal_ratios,
        "voxel_to_output_ratio": voxel_to_output_ratios,
        "reduction_ratio": reduction_ratios,
        "outside_points_ratio": outside_points_ratios
    })
    df.to_csv(os.path.join(output_dir, f"results_{param_name}.csv"), index=False)

def main():
    """Main function to run the parameter sweep."""
    args = parse_args()
    
    # Load input points
    print(f"Loading input file: {args.input}")
    points = load_xyz_file(args.input)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Define default parameters
    default_params = {
        "group_size": 1000,
        "voxel_x_size": 1.0,
        "voxel_y_size": 1.0,
        "anomaly_threshold": 0.4,  # Changed from 0.5 to match basic_usage.py
        "mode_probability_threshold": 3.0,  # Changed from 0.3 to match basic_usage.py
        "min_points_for_mode": 3,
        "max_modes": 2
    }
    
    # Determine which parameter is being swept
    param_name = None
    param_values = None
    
    if args.group_size:
        param_name = "group_size"
        param_values = [int(x) for x in args.group_size.split(",")]
    elif args.voxel_x_size:
        param_name = "voxel_x_size"
        param_values = [float(x) for x in args.voxel_x_size.split(",")]
        # Also update voxel_y_size to be the same
        default_params["voxel_y_size"] = default_params["voxel_x_size"]
    elif args.anomaly_threshold:
        param_name = "anomaly_threshold"
        param_values = [float(x) for x in args.anomaly_threshold.split(",")]
    elif args.mode_probability_threshold:
        param_name = "mode_probability_threshold"
        param_values = [float(x) for x in args.mode_probability_threshold.split(",")]
    elif args.min_points_for_mode:
        param_name = "min_points_for_mode"
        param_values = [int(x) for x in args.min_points_for_mode.split(",")]
    elif args.max_modes:
        param_name = "max_modes"
        param_values = [int(x) for x in args.max_modes.split(",")]
    else:
        print("Error: No parameter specified for sweeping")
        sys.exit(1)
    
    print(f"Starting parameter sweep for {param_name} with values: {param_values}")
    
    # Run parameter sweep
    results = []
    for param_value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        # Update parameters
        params = default_params.copy()
        if param_name == "voxel_x_size":
            params["voxel_x_size"] = param_value
            params["voxel_y_size"] = param_value
        else:
            params[param_name] = param_value
        
        # Run IsolationGrid and get subsampled points and stats
        start_time = time.time()
        subsampled_points, stats = run_isolation_grid(points, params)
        end_time = time.time()
        runtime = end_time - start_time
        
        # Calculate metrics
        # Reduction ratio is output_points/input_points
        reduction_ratio = len(subsampled_points) / len(points)
        
        # Extract voxel statistics using the correct keys from the stats dictionary
        total_voxels = stats['total_voxels']
        too_few_points = stats['voxels_too_few_points']
        unimodal = stats['voxels_unimodal']
        multimodal = stats['voxels_multimodal']
        high_anomaly_points = stats['high_anomaly_points']
        low_prob_points = stats['low_prob_points']
        
        # Calculate ratios
        high_anomaly_ratio = high_anomaly_points / (too_few_points + unimodal + multimodal) if (too_few_points + unimodal + multimodal) > 0 else 0
        std_unimodal_multimodal_ratio = (too_few_points + unimodal) / multimodal if multimodal > 0 else 0
        
        # Voxel to output ratio is total_voxels/total output points
        voxel_to_output_ratio = total_voxels / len(subsampled_points) if len(subsampled_points) > 0 else 0
        
        # Print debug information
        print(f"\nDebug info for {param_name}={param_value}:")
        print(f"Total voxels: {total_voxels}")
        print(f"Too few points voxels: {too_few_points}")
        print(f"Unimodal voxels: {unimodal}")
        print(f"Multimodal voxels: {multimodal}")
        print(f"Sum of voxel types: {too_few_points + unimodal + multimodal}")
        print(f"Output points: {len(subsampled_points)}")
        print(f"Voxel to output ratio: {voxel_to_output_ratio}")
        
        # Outside points ratio is low_prob_points/(too_few_points+unimodal+multimodal)
        outside_points_ratio = low_prob_points / (too_few_points + unimodal + multimodal) if (too_few_points + unimodal + multimodal) > 0 else 0
        
        # Save results
        result = {
            "param_name": param_name,
            "param_value": param_value,
            "runtime": runtime,
            "reduction_ratio": reduction_ratio,
            "high_anomaly_ratio": high_anomaly_ratio,
            "std_unimodal_multimodal_ratio": std_unimodal_multimodal_ratio,
            "voxel_to_output_ratio": voxel_to_output_ratio,
            "outside_points_ratio": outside_points_ratio,
            "total_voxels": total_voxels,
            "too_few_points": too_few_points,
            "unimodal": unimodal,
            "multimodal": multimodal,
            "high_anomaly_points": high_anomaly_points,
            "low_prob_points": low_prob_points
        }
        results.append(result)
        
        # Save subsampled points
        output_file = os.path.join(args.output, f"subsampled_{param_name}_{param_value}.xyz")
        save_xyz_file(subsampled_points, output_file)
    
    # Plot results
    plot_results(results, args.output, param_name)
    
    print(f"Parameter sweep completed. Results saved to {args.output}")

if __name__ == "__main__":
    main() 