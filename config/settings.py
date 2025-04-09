class Settings:
    """Configuration settings for the isolation grid processor."""
    
    def __init__(self, **kwargs):
        """Initialize settings with provided values or defaults."""
        self.group_size = kwargs.get('group_size', 1000)
        self.voxel_x_size = kwargs.get('voxel_x_size', 1.0)
        self.voxel_y_size = kwargs.get('voxel_y_size', 1.0)
        self.anomaly_threshold = kwargs.get('anomaly_threshold', 0.5)
        self.mode_probability_threshold = kwargs.get('mode_probability_threshold', 0.3)
        self.min_points_for_mode = kwargs.get('min_points_for_mode', 3)
        self.max_modes = kwargs.get('max_modes', 1)
        self.verbose = kwargs.get('verbose', False)
        self.save_intermediate_files = kwargs.get('save_intermediate_files', False)
        self.plot_interval = kwargs.get('plot_interval', 500)
