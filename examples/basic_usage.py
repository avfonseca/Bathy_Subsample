import os
import sys
import eif as iso

# Add the parent directory to Python path so we can import bathy_subsample
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from Bathy_Subsample import IsolationGrid

def main():
    """Example usage of IsolationGrid class."""
    processor = IsolationGrid(
        group_size=1000,
        voxel_x_size=3.0,
        voxel_y_size=3.0,
        mode_probability_threshold=2,
        anomaly_threshold=0.5,
        min_points_for_mode=3,
        max_modes=2,
        save_intermediate_files=False
    )
    
    final_points, stats = processor.process('example_points.xyz')

if __name__ == "__main__":
    main()
