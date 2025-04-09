from setuptools import setup, find_packages

setup(
    name="bathy_subsample",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "joblib>=1.0.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
        "eif>=2.0.0"
    ],
    python_requires=">=3.6",
    author="Adriano Fonseca",
    description="A Python package for processing bathymetric point clouds using isolation forests and voxel grids",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/avfonseca/bathy_subsample",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 