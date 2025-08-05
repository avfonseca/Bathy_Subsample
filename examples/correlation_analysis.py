import numpy as np
from scipy import stats
import os
import sys
from scipy.spatial.distance import pdist, squareform
import libpysal
from libpysal.weights import WSP, WSP2W
from esda.moran import Moran, Moran_Local
from esda.geary import Geary
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Add the parent directory to Python path so we can import bathy_subsample
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from Bathy_Subsample import IsolationGrid

def center_points(points):
    """Center points by removing the mean."""
    if len(points) == 0:
        return np.array([])
    
    # Center each coordinate separately
    centered = np.zeros_like(points)
    for i in range(3):  # x, y, z coordinates
        centered[:, i] = points[:, i] - np.mean(points[:, i])
    
    return centered

def main():
    # Initialize processor with parameters
    processor = IsolationGrid(
        group_size=1000,
        voxel_x_size=3.0,
        voxel_y_size=3.0,
        mode_probability_threshold=3,
        anomaly_threshold=0.5,
        min_points_for_mode=3,
        max_modes=3,
        save_intermediate_files=False
    )
    
    # Process points
    final_points, stats = processor.process('example_points.xyz')
    
    # Collect all distances
    all_xy_distances = []
    all_z_distances = []
    mode_info = []  # To store which mode each distance came from
    
    # Debug information
    print("\nAnalyzing point distributions:")
    print("Voxel size: 3.0m x 3.0m")
    print("Maximum possible distance within voxel: √(3² + 3²) = 4.24m")
    
    if 'mode_assignments' in stats:
        for voxel_key, modes in stats['mode_assignments'].items():
            for mode_idx, points in modes.items():
                if not points:  # Skip empty modes
                    continue
                    
                points = np.array(points)
                if points.ndim == 1:
                    points = points.reshape(-1, 3)
                
                # Skip if only one point (no distances to calculate)
                if len(points) < 2:
                    continue
                
                
                # Center points by removing mean
                centered_points = center_points(points)
                
                # Calculate pairwise distances for x,y coordinates
                xy_dist_matrix = squareform(pdist(centered_points[:, :2]))
                
                # Calculate pairwise distances for z coordinates
                z_dist_matrix = squareform(pdist(centered_points[:, 2].reshape(-1, 1)))
                
                # Get upper triangle of distance matrices (excluding diagonal)
                mask = np.triu_indices_from(xy_dist_matrix, k=1)
                xy_distances = xy_dist_matrix[mask]
                z_distances = z_dist_matrix[mask]
                
                # Skip if all distances are zero
                if np.all(xy_distances == 0) and np.all(z_distances == 0):
                    continue
                
                
                # Store distances and mode info
                all_xy_distances.extend(xy_distances)
                all_z_distances.extend(z_distances)
                mode_info.extend([f"Voxel{voxel_key}_Mode{mode_idx}"] * len(xy_distances))
    
    # Only create plot if we have distances to plot
    if all_xy_distances:
        # Convert lists to numpy arrays
        all_xy_distances = np.array(all_xy_distances)
        all_z_distances = np.array(all_z_distances)
        
        # Calculate correlation and covariance
        correlation = np.corrcoef(all_xy_distances, all_z_distances)[0,1]
        covariance = np.cov(all_xy_distances, all_z_distances)[0,1]
        
        # Create figure with 4 subplots (3 original + 1 zoomed)
        fig = plt.figure(figsize=(20, 5))
        
        # Plot 1: XY distances histogram
        ax1 = plt.subplot(141)
        ax1.hist(all_xy_distances, bins=50, alpha=0.5, label='XY distances', density=True)
        mean_xy = np.mean(all_xy_distances)
        std_xy = np.std(all_xy_distances)
        ax1.axvline(mean_xy, color='r', linestyle='--', label=f'Mean: {mean_xy:.3f}')
        ax1.axvline(mean_xy + std_xy, color='g', linestyle=':', label=f'Mean ± Std: {std_xy:.3f}')
        ax1.axvline(mean_xy - std_xy, color='g', linestyle=':')
        ax1.axvline(4.24, color='k', linestyle='--', label='Max possible (4.24m)')
        ax1.set_title('Distribution of Centered XY Distances')
        ax1.set_xlabel('Distance (meters)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Z distances histogram
        ax2 = plt.subplot(142)
        ax2.hist(all_z_distances, bins=50, alpha=0.5, label='Z distances', density=True)
        mean_z = np.mean(all_z_distances)
        std_z = np.std(all_z_distances)
        ax2.axvline(mean_z, color='r', linestyle='--', label=f'Mean: {mean_z:.3f}')
        ax2.axvline(mean_z + std_z, color='g', linestyle=':', label=f'Mean ± Std: {std_z:.3f}')
        ax2.axvline(mean_z - std_z, color='g', linestyle=':')
        ax2.set_title('Distribution of Centered Z Distances')
        ax2.set_xlabel('Distance (meters)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: 2D histogram of XY vs Z distances (full range)
        ax3 = plt.subplot(143)
        h_full = ax3.hist2d(all_xy_distances, all_z_distances, bins=100, density=True, cmap='viridis')
        plt.colorbar(h_full[3], ax=ax3, label='Density')
        ax3.set_title(f'Joint Probability Density\nXY: [0, {np.max(all_xy_distances):.1f}m], Z: [0, {np.max(all_z_distances):.1f}m]')
        ax3.set_xlabel('XY Distance (meters)')
        ax3.set_ylabel('Z Distance (meters)')
        ax3.grid(True, alpha=0.3)
        vmax = h_full[3].get_clim()[1]
        
        # Plot 4: 2D histogram of XY vs Z distances (zoomed)
        ax4 = plt.subplot(144)
        x_lim = (0, 50)
        z_lim = (0, 0.1)
        h_zoom = ax4.hist2d(all_xy_distances, all_z_distances, bins=100, 
                           density=True, cmap='viridis', range=[x_lim, z_lim], vmin=0, vmax=vmax)
        plt.colorbar(h_zoom[3], ax=ax4, label='Density')
        ax4.set_title(f'Zoomed Joint Probability Density\nXY: [0, {x_lim[1]:.1f}m], Z: [0, {z_lim[1]:.1f}m]')
        ax4.set_xlabel('XY Distance (meters)')
        ax4.set_ylabel('Z Distance (meters)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/mode_distances.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\nOverall Distance Statistics:")
        print(f"Number of modes analyzed: {len(set(mode_info))}")
        print(f"Total number of pairwise distances: {len(all_xy_distances)}")
        print("\nXY Distances:")
        print(f"Mean: {mean_xy:.3f}")
        print(f"Standard deviation: {std_xy:.3f}")
        print(f"Min: {np.min(all_xy_distances):.3f}")
        print(f"Max: {np.max(all_xy_distances):.3f}")
        print("\nZ Distances:")
        print(f"Mean: {mean_z:.3f}")
        print(f"Standard deviation: {std_z:.3f}")
        print(f"Min: {np.min(all_z_distances):.3f}")
        print(f"Max: {np.max(all_z_distances):.3f}")
        print(f"\nCorrelation between XY and Z distances: {correlation:.3f}")
        print(f"Covariance between XY and Z distances: {covariance:.3f}")
        print(f"\nPlot saved to {output_dir}/mode_distances.png")
        
        # --- Moran's I analysis on the full dataset ---
        print("\nPerforming Moran's I analysis on the full dataset (all points from all modes)...")
        # Gather all points from all modes
        all_points = []
        if 'mode_assignments' in stats:
            for voxel_key, modes in stats['mode_assignments'].items():
                for mode_idx, points in modes.items():
                    if not points:
                        continue
                    points = np.array(points)
                    if points.ndim == 1:
                        points = points.reshape(-1, 3)
                    all_points.append(points)
        if all_points:
            all_points = np.vstack(all_points)
            z = all_points[:, 2]
            n = len(z)
            EPSILON = 1e-10
            dist_matrix = cdist(all_points, all_points)
            dist_matrix = dist_matrix + EPSILON  # Add epsilon everywhere
            weights = 1.0 / dist_matrix
            np.fill_diagonal(weights, 0)  # Zero out self-weights
            # Row-standardize
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = np.divide(weights, row_sums, where=row_sums!=0)
            # Vectorized Moran's I
            z_mean = np.mean(z)
            z_dev = z - z_mean
            num = z_dev @ weights @ z_dev
            denom = np.sum(z_dev ** 2)
            moran_i = (n / np.sum(weights)) * (num / denom)
            print(f"Moran's I (full dataset, all points as neighbors): {moran_i:.4f}")
            
            # # Permutation test (vectorized)
            # n_permutations = 999
            # rng = np.random.default_rng()
            # z_permuted = np.array([rng.permutation(z) for _ in range(n_permutations)])  # shape (n_permutations, n)
            # z_permuted_mean = np.mean(z_permuted, axis=1, keepdims=True)
            # z_dev_permuted = z_permuted - z_permuted_mean  # shape (n_permutations, n)
            # # Vectorized numerator: (z_dev_permuted @ weights @ z_dev_permuted.T).diagonal()
            # num_permuted = np.einsum('ij,jk,ik->i', z_dev_permuted, weights, z_dev_permuted)
            # denom_permuted = np.sum(z_dev_permuted ** 2, axis=1)
            # moran_i_permuted = (n / np.sum(weights)) * (num_permuted / denom_permuted)
            # # p-value: proportion of permuted >= observed (for positive autocorrelation)
            # p_value = (np.sum(moran_i_permuted >= moran_i) + 1) / (n_permutations + 1)
            # print(f"Permutation test (n={n_permutations}): p-value = {p_value:.4f}")
            # # Optionally plot the null distribution
            # plt.figure(figsize=(7,4))
            # plt.hist(moran_i_permuted, bins=30, alpha=0.7, label='Null distribution')
            # plt.axvline(moran_i, color='r', linestyle='--', label=f'Observed Moran\'s I: {moran_i:.3f}')
            # plt.xlabel("Moran's I")
            # plt.ylabel('Frequency')
            # plt.title("Permutation Null Distribution of Moran's I")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f"{output_dir}/moransI_null_distribution.png", dpi=200)
            # plt.close()
            # print(f"Null distribution plot saved to {output_dir}/moransI_null_distribution.png")
        else:
            print("No points found for Moran's I analysis.")
    else:
        print("No valid distances found to plot.")

if __name__ == "__main__":
    main() 