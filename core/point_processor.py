import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict
import eif as iso
from sklearn.mixture import BayesianGaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning
from ..utils.visualization import Visualizer
import hashlib
import pickle
import os
from joblib import Parallel, delayed
warnings.filterwarnings('ignore', category=ConvergenceWarning) #We are already handling convergence warnings in the extract_gmm function

class PointProcessor:
    """Handles point cloud processing and leaf creation."""
    
    def __init__(self, settings):
        self.settings = settings
        self.voxel_processor = None  # Will be set by IsolationGrid
        self.total_voxels_processed = 0
        self.visualizer = Visualizer(settings)
        self.iforest_cache = {}  # Cache for isolation forest models
        self.cache_dir = None
        self.cache_hits = 0
        self.cache_attempts = 0
    
    def set_voxel_processor(self, voxel_processor):
        """Set the voxel processor instance."""
        self.voxel_processor = voxel_processor
    
    def set_cache_dir(self, cache_dir):
        """Set cache directory for model persistence."""
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, points):
        """Generate cache key based on point distribution statistics."""
        # Use statistical moments as cache key (much faster than full point comparison)
        stats = {
            'mean': np.mean(points, axis=0),
            'std': np.std(points, axis=0),
            'min': np.min(points, axis=0),
            'max': np.max(points, axis=0),
            'count': len(points)
        }
        # Create hash from statistics
        stats_str = str(sorted(stats.items()))
        return hashlib.md5(stats_str.encode()).hexdigest()
    
    def _load_cached_iforest(self, cache_key):
        """Load cached isolation forest model."""
        if not self.cache_dir or not self.settings.enable_caching:
            return None
        
        self.cache_attempts += 1
        
        cache_file = os.path.join(self.cache_dir, f"iforest_{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.cache_hits += 1
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_cached_iforest(self, cache_key, iforest):
        """Save isolation forest model to cache."""
        if not self.cache_dir or not self.settings.enable_caching:
            return
        
        # Check cache hit rate - disable if too low
        if self.cache_attempts > 10:  # Wait for some attempts first
            hit_rate = self.cache_hits / self.cache_attempts
            if hit_rate < 0.1:  # Less than 10% hit rate
                if self.settings.verbose:
                    print(f"Cache hit rate too low ({hit_rate:.1%}), disabling cache saves")
                return
        
        cache_file = os.path.join(self.cache_dir, f"iforest_{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(iforest, f)
        except:
            pass  # Silently fail if cache save fails
    
    def prepare_leaves(self, points, output_dir):
        """Prepare leaves for processing using Morton order (Z-order curve)."""
        n_points = len(points)
        if n_points == 0:
            return []
        
        # Compute Morton codes for spatial ordering
        morton_codes = self._compute_morton_codes(points)
        
        # Sort points by Morton code for spatial locality
        sorted_indices = np.argsort(morton_codes)
        
        # Create leaves by taking consecutive chunks
        leaf_data = []
        leaf_id = 0
        points_processed = 0
        
        for i in range(0, n_points, self.settings.group_size):
            chunk_indices = sorted_indices[i:i + self.settings.group_size]
            group_points = points[chunk_indices]
            points_processed += len(group_points)
            
            leaf_data.append((group_points, leaf_id, output_dir))
            leaf_id += 1
        
        return leaf_data
    
    def _compute_morton_codes(self, points):
        """Compute Morton codes (Z-order) for 3D points."""
        # Normalize coordinates to [0, 1023] range for 10-bit precision per dimension
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        ranges = max_coords - min_coords
        ranges = np.where(ranges == 0, 1, ranges)  # Avoid division by zero
        
        normalized = ((points - min_coords) / ranges * 1023).astype(np.uint32)
        normalized = np.clip(normalized, 0, 1023)  # Ensure within bounds
        
        # Compute Morton codes using bit interleaving
        morton_codes = np.zeros(len(points), dtype=np.uint64)
        
        for i in range(10):  # 10 bits per dimension
            bit = 1 << i
            morton_codes |= ((normalized[:, 0] & bit) << (2*i)) | \
                           ((normalized[:, 1] & bit) << (2*i + 1)) | \
                           ((normalized[:, 2] & bit) << (2*i + 2))
        
        return morton_codes


    def extract_all_modes(self, data):
        """Extract modes from depth values using Gaussian Mixture Model."""
        if len(data) == 0:
            return {'modes': [], 'gmm': None, 'type': 'empty'}
        
        # Calculate TVU for mean depth to use as threshold
        mean_depth = np.mean(data)
        tvu = self.voxel_processor.calculate_tvu(mean_depth)
        max_std = tvu / 2  # Use half TVU as threshold
        
        # Calculate overall statistics
        data_std = np.std(data)
        data_median = np.median(data)
        
        # If too few points or std is small enough, return median
        if len(data) < self.settings.min_points_for_mode:
                if self.settings.verbose:
                            print(f"Too few points or std is small enough")
                            print(f"Defaulting to standard deviation")

                return {
                    'modes': [{'mean': data_median, 'std': min(data_std, max_std)}],
                    'gmm': None,
                    'type': "std" 
                }
        
            
        
        # Try fitting GMM with N modes
        n_modes = self.settings.max_modes
        
        if (len(data)  >=  self.settings.min_points_for_mode):
            try:
                try_modes = min(n_modes,len(data)//self.settings.min_points_for_mode)
                result = self.extract_gmm(data, try_modes, best_hypothesis= self.settings.best_hypothesis)
                if self.settings.verbose:
                    print(f"n_modes: {try_modes}, GMM converged: {result['gmm'].converged_}, n_components: {result['gmm'].n_components}")
                return result
            
            
            except Exception as e:
                if self.settings.verbose:
                    print(f"GMM fitting failed Defaulting to standard deviation: {str(e)}")
                    
                return {
                        'modes': [{'mean': data_median, 'std': min(data_std, max_std)}],
                        'gmm': None,
                        'type': "std" 
                    }


    def extract_gmm(self, data, n_modes, best_hypothesis=False):
        gmm = BayesianGaussianMixture(n_components=n_modes,
                                         covariance_type='spherical',
                                random_state=42,
                                max_iter=50,  # Reduced from 100
                                n_init=3,     # Reduced from 5
                                init_params='k-means++')
            
        # Fit GMM
        gmm.fit(data.reshape(-1, 1))
        
        # Extract modes and sort by mean
        modes = []


        if best_hypothesis:
            largest_weight_idx = np.argmax(gmm.weights_)
            modes.append({'mean': gmm.means_[largest_weight_idx][0], 'std': np.sqrt(gmm.covariances_[largest_weight_idx].flatten()[0])})
        else:
            for mean, covar in zip(gmm.means_, gmm.covariances_):
                std = np.sqrt(covar.flatten()[0])
                modes.append({'mean': mean[0], 'std': std})
        
        # Sort modes by mean depth
        modes.sort(key=lambda x: x['mean'])
        
        
        return {
            'modes': modes,
            'gmm': gmm,
            'type': "unimodal" if gmm.n_components == 1 else "multimodal"
        }


        



    def analyze_group_with_iforest(self, points, leaf_id, output_dir):
        """Analyze a group of points with isolation forest."""
        # Create isolation forest with normalized points
        points_for_iforest = points.copy()
        points_mean = np.mean(points_for_iforest, axis=0)
        points_std = np.std(points_for_iforest, axis=0)
        points_for_iforest = (points_for_iforest - points_mean) / points_std
        
        # Try to load cached model first (only if caching is enabled)
        iforest = None
        if self.settings.enable_caching:
            cache_key = self._get_cache_key(points_for_iforest)
            iforest = self._load_cached_iforest(cache_key)
        
        if iforest is None:
            # Create new model if not cached
            sample_size = min(128, len(points))  # Reduced from 256
            iforest = iso.iForest(points_for_iforest, ntrees=100, sample_size=sample_size)  # Reduced from 200
            
            # Cache the model for future use (only if caching is enabled)
            if self.settings.enable_caching:
                cache_key = self._get_cache_key(points_for_iforest)
                self._save_cached_iforest(cache_key, iforest)
        
        scores = iforest.compute_paths(X_in=points_for_iforest)
        
        # Normalize scores to [0,1] range
        score_range = np.max(scores) - np.min(scores)
        if score_range > 0:
            scores_normalized = (scores - np.min(scores)) / score_range
        else:
            # If all scores are the same, set all normalized scores to 0.5
            scores_normalized = np.full_like(scores, 0.5)
        
        # Create voxel grid and get transformation matrices
        (voxel_indices, x_size, y_size, z_size, rotation_matrix, 
         points_aligned, avg_normal, mean_point) = self.voxel_processor.create_voxel_grid(points)
        
        # Group points by voxel
        voxel_dict = defaultdict(list)
        for i, (point, score) in enumerate(zip(points_aligned, scores_normalized)):
            voxel_key = tuple(voxel_indices[i])
            voxel_dict[voxel_key].append((point, score, i))
        
        # Initialize lists for different point categories
        selected_points = []
        high_anomaly_points = []
        mode_representatives = []  # List of points that represent each mode
        low_prob_points = []
        point_strengths = []
        
        # Initialize counters
        n_high_anomaly = 0
        n_extracted_modes = 0
        n_outliers = 0
        # Track voxel categories
        n_voxels_std = 0
        n_voxels_unimodal = 0
        n_voxels_multimodal = 0
       
        total_voxels = len(voxel_dict)
        
        # Track mode assignments for all voxels
        all_mode_assignments = {}  # Dictionary of all points assigned to each mode
        
        # Process voxels in parallel instead of sequentially
        parallel_results = self._process_voxels_parallel(voxel_dict, rotation_matrix, mean_point)
        
        # Process results from parallel computation
        for processed_data in parallel_results:
            voxel_key = processed_data['voxel_key']
            result = processed_data['result']
            low_anomaly_points = processed_data['low_anomaly_points']
            voxel_scores = processed_data['voxel_scores']
            voxel_points_array = processed_data['voxel_points_array']
            high_anomaly_points_voxel = processed_data['high_anomaly_points']
            
            # Handle high anomaly points from parallel processing
            if len(high_anomaly_points_voxel) > 0:
                high_anomaly_points.extend(high_anomaly_points_voxel)
                if self.settings.navigation == False:
                    selected_points.extend(high_anomaly_points_voxel)
                    point_strengths.extend([1] * len(high_anomaly_points_voxel))
                n_high_anomaly += len(high_anomaly_points_voxel)
            
            # Track voxel category
            if result['type'] == "std":
                n_voxels_std += 1
            elif result['type'] == "unimodal":
                n_voxels_unimodal += 1
            elif result['type'] == "multimodal":
                n_voxels_multimodal += 1

            if result['type'] == "std" or result['type'] == "unimodal":
                median = result['modes'][0]['mean']
                std_th = result['modes'][0]['std']
                n_extracted_modes += 1

                # Track points belonging to mode (in transformed space)
                mode_points_dict = {0: []}  # Single mode with index 0
                
                # Calculate z-scores for all points
                z_scores = abs(low_anomaly_points[:, 2] - median)/(std_th + 1e-10)
                
                # Store points within threshold
                mode_mask = z_scores <= self.settings.mode_probability_threshold
                mode_points_voxel = low_anomaly_points[mode_mask]
                mode_points_dict[0] = mode_points_voxel.tolist()  # Store ALL points in this mode
                
                # Store mode assignments
                all_mode_assignments[voxel_key] = mode_points_dict
                
                # Find and store the representative point for this mode
                median_point = min(low_anomaly_points, key=lambda p: abs(p[2] - median))
                median_point_original = np.dot(median_point.reshape(1, -1), rotation_matrix.T) + mean_point
                mode_representatives.append(median_point_original)  # Store the representative point
                selected_points.append(median_point_original[0])
                
                # Keep points outside std limit
                mode_str = len(mode_points_dict[0])
                point_strengths.append(mode_str)

                outside_points = low_anomaly_points[~mode_mask]
                
                if len(outside_points) > 0:
                    outside_points_original = np.dot(outside_points, rotation_matrix.T) + mean_point
                    low_prob_points.extend(outside_points_original)  # Track these points
                    selected_points.extend(outside_points_original)
                    n_outliers += len(outside_points)
                    point_strengths.extend([1] * len(outside_points))
            
            elif result['type'] == "multimodal": # N > 1
                medians = np.array([mode['mean'] for mode in result['modes']]).reshape(-1, 1)
                std_ths = np.array([mode['std'] for mode in result['modes']]).reshape(-1, 1)

                # Reshape depths for broadcasting
                depths = low_anomaly_points[:, 2].reshape(1, -1)  # Shape: (1, n_points)

                # Calculate normalized distances for each point to each mode
                normalized_distances = abs(depths - medians) / std_ths  # Shape: (n_modes, n_points)
                
                # Initialize array to track which mode each point belongs to
                point_mode_assignments = np.full(len(low_anomaly_points), -1)  # -1 means no mode assigned
                
                # Track points belonging to each mode (in transformed space)
                mode_points_dict = {i: [] for i in range(len(medians))}
                
                # For each point, find the closest mode
                for i in range(len(low_anomaly_points)):
                    # Get normalized distances to all modes for this point
                    point_distances = normalized_distances[:, i]
                    
                    # Find the closest mode
                    closest_mode_idx = np.argmin(point_distances)
                    
                    # Check if the point is within threshold of the closest mode
                    if point_distances[closest_mode_idx] <= self.settings.mode_probability_threshold:
                        point_mode_assignments[i] = closest_mode_idx
                        # Store the point in transformed space
                        mode_points_dict[closest_mode_idx].append(low_anomaly_points[i].tolist())
                
                # Store mode assignments for this voxel
                all_mode_assignments[voxel_key] = mode_points_dict
                
                # Count points belonging to each mode
                mode_counts = np.zeros(len(medians), dtype=int)
                for mode_idx in range(len(medians)):
                    mode_counts[mode_idx] = len(mode_points_dict[mode_idx])
                
                # Add mode representative points
                for i in range(len(medians)):
                    # Only add representative points for modes that actually have points assigned
                    if mode_counts[i] > 0:
                        # Find the point closest to this mode's median
                        median_point = min(low_anomaly_points, key=lambda p: abs(p[2] - medians[i]))
                        median_point_original = np.dot(median_point.reshape(1, -1), rotation_matrix.T) + mean_point
                        mode_representatives.append(median_point_original)  # Store the representative point
                        selected_points.append(median_point_original[0])
                        point_strengths.append(mode_counts[i])
                        n_extracted_modes += 1
                
                # Points that don't belong to any mode
                outside_mask = point_mode_assignments == -1
                outside_points = low_anomaly_points[outside_mask]
                
                if len(outside_points) > 0 and self.settings.navigation == False:
                    outside_points_original = np.dot(outside_points, rotation_matrix.T) + mean_point
                    low_prob_points.extend(outside_points_original)  # Track these points
                    selected_points.extend(outside_points_original)
                    n_outliers += len(outside_points)
                    point_strengths.extend([1] * len(outside_points))
                
               
        # Convert lists to numpy arrays
        selected_points = np.array(selected_points)
        high_anomaly_points = np.array(high_anomaly_points) if high_anomaly_points else np.empty((0, 3))
        mode_representatives = np.vstack(mode_representatives) if mode_representatives else np.empty((0, 3))
        low_prob_points = np.array(low_prob_points) if low_prob_points else np.empty((0, 3))
        point_strengths = np.array(point_strengths) if point_strengths else np.empty((0,))

        if self.settings.save_intermediate_files:
            self.visualizer.visualize_leaf(points, leaf_id, output_dir, 
                                        selected_points, high_anomaly_points, 
                                        mode_representatives, low_prob_points, point_strengths)
        
        # Return both points and voxel statistics
        voxel_stats = {
            'total_voxels': n_voxels_multimodal + n_voxels_unimodal + n_voxels_std,
            'too_few_points': n_voxels_std,
            'unimodal': n_voxels_unimodal,
            'multimodal': n_voxels_multimodal,
            'high_anomaly_points': n_high_anomaly,
            'low_prob_points': len(low_prob_points),
            'mode_points': len(mode_representatives),
            'point_strengths': point_strengths,
            'mode_assignments': all_mode_assignments
        }
        
        return selected_points, voxel_stats

    def print_cache_stats(self):
        """Print cache statistics if verbose is enabled."""
        if self.settings.verbose and self.cache_attempts > 0:
            hit_rate = self.cache_hits / self.cache_attempts
            print(f"\nCache Statistics:")
            print(f"  Cache attempts: {self.cache_attempts}")
            print(f"  Cache hits: {self.cache_hits}")
            print(f"  Hit rate: {hit_rate:.1%}")
            if hit_rate < 0.1 and self.cache_attempts > 10:
                print(f"  Warning: Low cache hit rate - consider disabling caching")
    
    def _process_voxel_gmm_parallel(self, voxel_data):
        """Process a single voxel with GMM fitting (for parallel processing)."""
        voxel_key, voxel_points, voxel_scores, rotation_matrix, mean_point = voxel_data
        
        # Unpack points
        voxel_points_array = np.array([p[0] for p in voxel_points])
        
        if len(voxel_points_array) == 0:
            return None
        
        # Get high anomaly points
        high_anomaly_mask = voxel_scores > self.settings.anomaly_threshold
        high_anomaly_points_voxel = voxel_points_array[high_anomaly_mask]
        
        # Get low anomaly points
        low_anomaly_mask = voxel_scores <= self.settings.anomaly_threshold
        low_anomaly_points = voxel_points_array[low_anomaly_mask]
        
        # Extract modes from depth values
        result = self.extract_all_modes(low_anomaly_points[:, 2]) if len(low_anomaly_points) > 0 else {'modes': [], 'type': 'empty'}
        
        # Process based on result type
        processed_data = {
            'voxel_key': voxel_key,
            'high_anomaly_points': np.dot(high_anomaly_points_voxel, rotation_matrix.T) + mean_point if len(high_anomaly_points_voxel) > 0 else np.empty((0, 3)),
            'result': result,
            'low_anomaly_points': low_anomaly_points,
            'rotation_matrix': rotation_matrix,
            'mean_point': mean_point,
            'voxel_scores': voxel_scores,
            'voxel_points_array': voxel_points_array
        }
        
        return processed_data
    
    def _process_voxels_parallel(self, voxel_dict, rotation_matrix, mean_point):
        """Process all voxels in parallel using GMM fitting."""
        # Prepare data for parallel processing
        voxel_data_list = []
        for voxel_key, voxel_points in voxel_dict.items():
            voxel_scores = np.array([p[1] for p in voxel_points])
            voxel_data_list.append((voxel_key, voxel_points, voxel_scores, rotation_matrix, mean_point))
        
        # Process voxels in parallel
        n_jobs = min(8, len(voxel_data_list))  # Limit to 8 jobs to avoid memory issues
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(self._process_voxel_gmm_parallel)(voxel_data) 
            for voxel_data in voxel_data_list
        )
        
        # Filter out None results
        return [r for r in results if r is not None]
