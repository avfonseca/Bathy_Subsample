import numpy as np

class StatsCollector:
    """Handles statistics collection and reporting."""
    
    def combine_results(self, all_processed_results, original_points, output_dir):
            """Combine results from all processed leaves."""
            # Separate points and statistics
            all_processed_points = [result[0] for result in all_processed_results]
            all_voxel_stats = [result[1] for result in all_processed_results]

            # Combine points
            final_points = np.vstack(all_processed_points)
            np.savetxt(f"{output_dir}/processed_points.xyz", final_points)

            # Combine statistics
            original_point_count = len(original_points)
            final_point_count = len(final_points)
            reduction_percentage = ((original_point_count - final_point_count) / original_point_count) * 100

            stats = {
            'original_points': original_point_count,
            'processed_points': final_point_count,
            'points_reduced': original_point_count - final_point_count,
            'reduction_percentage': reduction_percentage,
            'total_voxels': sum(s['total_voxels'] for s in all_voxel_stats),
            'voxels_too_few_points': sum(s['too_few_points'] for s in all_voxel_stats),
            'voxels_unimodal': sum(s['unimodal'] for s in all_voxel_stats),
            'voxels_multimodal': sum(s['multimodal'] for s in all_voxel_stats),
            'high_anomaly_points': sum(s['high_anomaly_points'] for s in all_voxel_stats),
            'low_prob_points': sum(s.get('low_prob_points', 0) for s in all_voxel_stats)
            }

                  
            return final_points, stats

    def print_summary_stats(self, stats):
            """Print summary statistics."""
            print("\nPoint Cloud Processing Results:")
            print(f"Input points:  {stats['original_points']:,}")
            print(f"Output points: {stats['processed_points']:,}")
            print(f"Reduction:     {stats['reduction_percentage']:.1f}%")
            
            print("\nVoxel Category Breakdown:")
            print(f"Total voxels: {stats['total_voxels']:,}")
            print(f"Too few points: {stats['voxels_too_few_points']:,} "
                  f"({(stats['voxels_too_few_points']/stats['total_voxels'])*100:.1f}%)")
            print(f"Unimodal:      {stats['voxels_unimodal']:,} "
                  f"({(stats['voxels_unimodal']/stats['total_voxels'])*100:.1f}%)")
            print(f"Multimodal:    {stats['voxels_multimodal']:,} "
                  f"({(stats['voxels_multimodal']/stats['total_voxels'])*100:.1f}%)")
            
            print("\nPoint Type Breakdown:")
            print(f"Mode/median points: "
                  f"{stats['processed_points'] - stats['high_anomaly_points'] - stats['low_prob_points']:,}")
            print(f"High anomaly points: {stats['high_anomaly_points']:,}")
            print(f"Low prob/outside std: {stats['low_prob_points']:,}")
