import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Visualizer:
    """Handles all visualization tasks."""
    
    def __init__(self, settings):
        self.settings = settings
    
    def visualize_points(self, points, scores=None, title="Point Cloud", 
                        output_file=None, s=1, alpha=1.0):
        """Visualize points in 3D with optional coloring by scores."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if scores is not None:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=scores, cmap='viridis', s=s, alpha=alpha)
            plt.colorbar(scatter, label='Anomaly Score')
        else:
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c='blue', s=s, alpha=alpha)
        
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_leaf(self, points, leaf_id, output_dir, selected_points, 
                      high_anomaly_points, mode_points, low_prob_points):
        """Visualize a leaf of points with three subplots."""
        fig = plt.figure(figsize=(20, 6))
        
        # Original point cloud subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c='blue', s=5, alpha=0.7)
        ax1.set_title(f'Leaf {leaf_id} - Original Point Cloud\n{len(points):,} points')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # Selected points subplot
        ax2 = fig.add_subplot(132, projection='3d')
        
        if len(high_anomaly_points) > 0:
            ax2.scatter(high_anomaly_points[:, 0], high_anomaly_points[:, 1], 
                       high_anomaly_points[:, 2], c='red', s=5, alpha=0.7, 
                       label='High Anomaly')
        
        if len(mode_points) > 0:
            ax2.scatter(mode_points[:, 0], mode_points[:, 1], mode_points[:, 2],
                       c='blue', s=5, label='Mode/Median')
        
        if len(low_prob_points) > 0:
            ax2.scatter(low_prob_points[:, 0], low_prob_points[:, 1], 
                       low_prob_points[:, 2], c='green', s=5, alpha=0.7, 
                       label='Low Prob/Outside')
        
        title_str = f'Leaf {leaf_id} - Selected Points\n'
        title_str += f'High Anomaly: {len(high_anomaly_points):,}\n'
        title_str += f'Mode/Median: {len(mode_points):,}\n'
        title_str += f'Low Prob/Outside: {len(low_prob_points):,}'
        
        ax2.set_title(title_str)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        ax2.legend()
        
        # Depth subplot
        ax3 = fig.add_subplot(133, projection='3d')
        scatter = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                            c=points[:, 2], cmap='viridis', s=5, alpha=0.7)
        plt.colorbar(scatter, label='Z (m)')
        
        ax3.set_title(f'Leaf {leaf_id} - Point Depths')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/leaf_{leaf_id}_visualization.png", dpi=300, 
                   bbox_inches='tight')
        plt.close()

    def create_summary_plots(self, original_points, processed_points, stats, output_dir):
        """Create summary visualization plots."""
        reduction_percentage = stats['reduction_percentage']
        
        self.visualize_points(original_points,
                            title='Original Complete Point Cloud',
                            output_file=f"{output_dir}/original_complete.png")
        
        self.visualize_points(processed_points,
                            title=f'Processed Complete Point Cloud\n{reduction_percentage:.1f}% reduction',
                            output_file=f"{output_dir}/processed_complete.png")
