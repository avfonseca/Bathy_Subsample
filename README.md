# Bathy Subsample

A Python package for processing bathymetric point clouds using isolation forests and voxel grids. This tool helps in subsampling bathymetric data while preserving important features and adhering to IHO standards.

## Features

- Point cloud anomaly detection using isolation forests
- Adaptive voxel grid processing
- Multi-modal depth analysis
- TVU-based uncertainty handling
- Parallel processing support
- Comprehensive visualization tools
- IHO standards compliance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bathy_subsample.git
cd bathy_subsample

# Install in development mode
pip install -e .
```

## Requirements

- Python 3.6+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- joblib
- tqdm
- seaborn

## Usage

Basic usage example:

```python
from bathy_subsample import IsolationGrid

# Create processor instance
processor = IsolationGrid(
    group_size=1000,
    voxel_x_size=1.0,
    voxel_y_size=1.0,
    mode_probability_threshold=0.6,
    anomaly_threshold=0.5,
    min_points_for_mode=5,
    max_modes=2,
    save_intermediate_files=True
)

# Process point cloud
final_points, stats = processor.process('input_points.xyz')
```

## Configuration Parameters

- `group_size`: Number of points to process in each leaf (default: 1000)
- `voxel_x_size`: Size of voxels in X dimension in meters (default: 1.0)
- `voxel_y_size`: Size of voxels in Y dimension in meters (default: 1.0)
- `anomaly_threshold`: Threshold for anomaly scores (default: 0.5)
- `mode_probability_threshold`: Minimum probability to assign point to a mode
- `min_points_for_mode`: Minimum points needed for mode fitting
- `max_modes`: Maximum number of modes to fit (default: 1)

## Output

The processor generates:
- Processed point cloud file
- Visualization plots
- Statistical analysis
- Processing reports

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{bathy_subsample,
  author = {Adriano Fonseca},
  title = {Bathy Subsample: A Tool for Bathymetric Point Cloud Processing},
  year = {2024},
  url = {https://github.com/avfonseca/bathy_subsample}
}
```
```

