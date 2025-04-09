import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import sys

from ..utils.visualization import Visualizer
from ..utils.statistics import StatsCollector
from .voxel_processor import VoxelProcessor
from .point_processor import PointProcessor
from ..config.settings import Settings

# Define version directly if needed
__version__ = "1.0.0"

class IsolationGrid:
    """
    A class for processing point clouds using isolation forests and voxel grids.
    
    This class implements a method for denoising and subsampling point clouds by:
    1. Dividing the point cloud into leaves of nearest neighbors
    2. Using isolation forests to detect anomalies in each leaf
    3. Creating a voxel grid for each leaf
    4. Processing points within each voxel based on anomaly scores and mode fitting
    """
    
    def __init__(self, group_size=1000, voxel_x_size=1.0, voxel_y_size=1.0, 
                 anomaly_threshold=0.5, mode_probability_threshold=0.3,
                 min_points_for_mode=3, max_modes=1, verbose=False, 
                 save_intermediate_files=False, plot_interval=500):
        """Initialize the IsolationGrid processor."""
        self.settings = Settings(
            group_size=group_size,
            voxel_x_size=voxel_x_size,
            voxel_y_size=voxel_y_size,
            anomaly_threshold=anomaly_threshold,
            mode_probability_threshold=mode_probability_threshold,
            min_points_for_mode=min_points_for_mode,
            max_modes=max_modes,
            verbose=verbose,
            save_intermediate_files=save_intermediate_files,
            plot_interval=plot_interval
        )
        
        self.visualizer = Visualizer(self.settings)
        self.stats_collector = StatsCollector()
        self.voxel_processor = VoxelProcessor(self.settings)
        self.point_processor = PointProcessor(self.settings)
        self.point_processor.set_voxel_processor(self.voxel_processor)

    def process_leaf(self, leaf_data):
        """Process a single leaf of points."""
        points, leaf_id, output_dir = leaf_data
        return self.point_processor.analyze_group_with_iforest(points, leaf_id, output_dir)

    def process(self, input_file, output_dir="isolation_forest_results", n_jobs=-1):
        """Process a point cloud file using the isolation grid method."""
        self.point_processor.total_voxels_processed = 0
        os.makedirs(output_dir, exist_ok=True)
        
        # Load point cloud
        print("Loading point cloud...")
        points = np.loadtxt(input_file)
        
        # Prepare leaves for processing
        leaf_data = self.point_processor.prepare_leaves(points, output_dir)
        
        # Process leaves in parallel with progress bar
        print("\nProcessing leaves...")
        with tqdm(total=len(leaf_data), desc="Processing leaves", 
                 disable=not self.settings.verbose) as pbar:
            def process_with_progress(leaf):
                result = self.process_leaf(leaf)
                pbar.update(1)
                return result
            
             # Use context manager for proper cleanup
            with Parallel(n_jobs=n_jobs) as parallel:
                all_processed_results = parallel(
                delayed(process_with_progress)(leaf) for leaf in leaf_data
            )
        
        if self.settings.verbose:
            print()  # Add newline after progress bar
        
        # Combine and save results
        final_points, stats = self.stats_collector.combine_results(
            all_processed_results, points, output_dir)
        
        # Create visualizations if requested
        if self.settings.save_intermediate_files:
            self.visualizer.create_summary_plots(points, final_points, stats, output_dir)
        
        # Print statistics
        self.stats_collector.print_summary_stats(stats)
        
        return final_points, stats
