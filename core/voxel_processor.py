import numpy as np
from sklearn.neighbors import KDTree

class VoxelProcessor:
    """Handles voxel grid creation and processing."""
    
    def __init__(self, settings):
        self.settings = settings
    
    def calculate_tvu(self, depth, a=0.15, b=0.0075):
        """Calculate Total Vertical Uncertainty for a given depth."""
        return np.sqrt(a**2 + (b * depth)**2)
    
    def estimate_normals(self, points, k=20):
        """Estimate normal vectors using points within ±3*TVU of median depth."""
        # Get median depth and TVU
        median_depth = np.median(points[:, 2])
        tvu = self.calculate_tvu(median_depth)
        
        # Select points within ±3*TVU of median depth
        depth_mask = np.abs(points[:, 2] - median_depth) <= 3 * tvu
        points_for_normal = points[depth_mask]
        
        if self.settings.verbose:
            print(f"Using {np.sum(depth_mask)} points within ±3*TVU of median depth "
                  f"({median_depth:.2f}m) for normal estimation")
        
        # Ensure we have enough points
        k = min(k, len(points_for_normal) - 1)
        if k < 3:
            if self.settings.verbose:
                print("Warning: Not enough points for normal estimation, defaulting to vertical")
            return np.array([0, 0, 1]), depth_mask
        
        # Build KD-tree for nearest neighbor search
        tree = KDTree(points_for_normal)
        
        # Find k nearest neighbors for each point
        distances, indices = tree.query(points_for_normal, k=k)
        
        # Compute normal for each point using PCA
        normals = []
        for i in range(len(points_for_normal)):
            neighbors = points_for_normal[indices[i]]
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = np.dot(centered.T, centered)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            normal = eigenvecs[:, 0]  # Smallest eigenvector is normal
            # Ensure normal points "up" (positive Z)
            if normal[2] < 0:
                normal = -normal
            normals.append(normal)
        
        # Average the normals
        avg_normal = np.mean(normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)  # Normalize
        
        return avg_normal, depth_mask

    def create_voxel_grid(self, points):
        """Create a voxel grid aligned with surface normal."""
        # Calculate average normal and create transformation matrix
        avg_normal, depth_mask = self.estimate_normals(points)
        
        # Create orthonormal basis with avg_normal as Z axis
        z_axis = avg_normal
        x_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(x_axis, z_axis)) > 0.9:
            x_axis = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Get mean point for centering
        mean_point = np.mean(points, axis=0)
        
        # Transform points to align with normal
        points_centered = points - mean_point
        points_aligned = np.dot(points_centered, rotation_matrix)
        
        # Calculate voxel sizes in meters
        mean_depth = np.mean(points[:, 2])
        z_size = self.calculate_tvu(mean_depth)  # TVU for Z
        
        # Create grid indices
        x_idx = np.floor(points_aligned[:, 0] / self.settings.voxel_x_size).astype(int)
        y_idx = np.floor(points_aligned[:, 1] / self.settings.voxel_y_size).astype(int)
        z_idx = np.floor(points_aligned[:, 2] / z_size).astype(int)
        
        # Combine indices into single array
        voxel_indices = np.column_stack((x_idx, y_idx, z_idx))
        
        # Get unique voxels and counts
        unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)
        
        if self.settings.verbose:
            print(f"Created grid with {len(unique_voxels)} voxels")
            print(f"Voxel dimensions: X={self.settings.voxel_x_size:.3f}m, "
                  f"Y={self.settings.voxel_y_size:.3f}m, Z={z_size:.3f}m (TVU)")
        
        return (voxel_indices, self.settings.voxel_x_size, 
                self.settings.voxel_y_size, z_size, rotation_matrix, 
                points_aligned, avg_normal, mean_point)
