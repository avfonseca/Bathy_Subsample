import os
import sys
# Import version first
from version import VERSION, __version__
import eif as iso
print(f"Loading eif module from: {os.path.abspath(iso.__file__)}")

# Get the absolute path to the project root directory (two levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from bathy_subsample import IsolationGrid

def main():
    """Example usage of IsolationGrid class."""
    processor = IsolationGrid(
        group_size=1000,
        voxel_x_size=1.0,
        voxel_y_size=1.0,
        mode_probability_threshold=2,
        anomaly_threshold=0.5,
        min_points_for_mode=3,
        max_modes=2,
        save_intermediate_files=True
    )
    
    final_points, stats = processor.process('example_points.xyz')

if __name__ == "__main__":
    main()
