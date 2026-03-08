# GeoAttNet: Geophysical and Geochemical Attention Network for Mineral Exploration

This repository contains the implementation of GeoAttNet, a deep learning framework for geophysical and geochemical data analysis and mineral exploration. The model is based on DeepUNet architecture and enhanced with attention mechanisms (CBAM) for processing multi-channel geophysical and geochemical raster data to predict mineral occurrence probabilities.

## Repository Structure

```
├── GeoAttNet/                    # Main GeoAttNet implementation
│   ├── GeoAttNet_model.py        # Core U-Net model with attention mechanisms
│   ├── data_selection.py         # Geophysical/geochemical data processing and patch extraction
│   ├── standardize_for_prediction.py  # Data standardization for inference
│   ├── test_model.py             # Model testing and prediction pipeline
│   └── train_GeoAttNet.py        # Training script with combined loss functions
├── GeoAttNet-Base/               # Baseline version without attention
│   ├── GeoAttNet_model.py        # U-Net model without attention mechanisms
│   ├── data_selection.py         # Data processing utilities
│   ├── standardize_for_prediction.py  # Standardization utilities
│   ├── test_model.py             # Testing and evaluation
│   └── train_GeoAttNet.py        # Training with standard BCE loss
├── GeoAttNet-CBAM/               # Enhanced version with CBAM attention
│   ├── GeoAttNet_model.py        # U-Net with Channel and Spatial Attention
│   ├── data_selection.py         # Data processing with attention support
│   ├── standardize_for_prediction.py  # Enhanced standardization
│   ├── test_model.py             # Advanced testing pipeline
│   └── train_GeoAttNet.py        # Training with attention mechanisms
├── GeoAttNet-Loss/               # Version with advanced loss functions
│   ├── GeoAttNet_model.py        # U-Net with optimized architecture
│   ├── data_selection.py         # Enhanced data selection
│   ├── standardize_for_prediction.py  # Advanced standardization
│   ├── test_model.py             # Comprehensive testing
│   └── train_GeoAttNet.py        # Training with Focal + Dice loss
├── missing_value/                # Geophysical/geochemical data interpolation module
│   ├── geochemical_interpolation.py  # Advanced interpolation algorithms
│   ├── missing_value_analysis.py     # Missing data pattern analysis
│   └── run_interpolation.py          # Automated interpolation pipeline
├── compare_training_curves_simple.py # Training performance comparison
├── fix_stats_key.py              # Statistical data correction utilities
└── README.md                     # Project documentation
```


## Features

- **DeepUNet-based Architecture**: Enhanced encoder-decoder network optimized for geophysical and geochemical data analysis
- **Attention Mechanisms**: CBAM (Channel and Spatial Attention) for enhanced feature extraction
- **Advanced Loss Functions**: Combined Focal and Dice loss for handling class imbalance
- **Geospatial Data Processing**: Automated raster data alignment and patch extraction
- **Missing Value Interpolation**: Intelligent interpolation based on spatial patterns
- **Multi-variant Comparison**: Four different model configurations for performance analysis

## Installation

1. Install dependencies:

   ```bash
   pip install torch torchvision rasterio geopandas scikit-learn pandas numpy matplotlib seaborn tqdm
   ```

2. Prepare your geophysical and geochemical dataset:

   - Place geophysical and geochemical raster files (.tif) in the data directory
   - Ensure mineral occurrence points are in GeoPackage (.gpkg) format
   - Verify coordinate reference systems are consistent

## Usage

### Data Preprocessing and Missing Value Interpolation

Before training, process missing values in geophysical and geochemical data:

```bash
# Analyze missing value patterns
python missing_value/missing_value_analysis.py

# Run automated interpolation
python missing_value/run_interpolation.py
```

### Training

Choose one of the four model variants and train:

```bash
# Train baseline model (no attention)
python GeoAttNet-Base/train_GeoAttNet.py

# Train model with CBAM attention
python GeoAttNet-CBAM/train_GeoAttNet.py

# Train model with advanced loss functions
python GeoAttNet-Loss/train_GeoAttNet.py

# Train main GeoAttNet model
python GeoAttNet/train_GeoAttNet.py
```

### Prediction and Testing

Generate mineral occurrence probability maps:

```bash
# Test and predict with trained model
python GeoAttNet/test_model.py

# Compare different model variants
python compare_training_curves_simple.py
```

## Model Variants

### GeoAttNet (Main)
The primary implementation featuring:
- DeepUNet architecture with skip connections
- CBAM attention mechanisms (Channel + Spatial)
- Combined Focal and Dice loss functions
- Optimized for 15-channel geophysical/geochemical input (32×32 patches)

### GeoAttNet-Base
Baseline version for comparison:
- Standard DeepUNet without attention mechanisms
- Binary Cross-Entropy loss
- Simplified architecture for performance benchmarking

### GeoAttNet-CBAM
Enhanced attention-focused variant:
- Advanced CBAM implementation
- Channel attention with squeeze-and-excitation
- Spatial attention with max and average pooling
- Improved feature representation

### GeoAttNet-Loss
Advanced loss function variant:
- Focal Loss for handling class imbalance
- Dice Loss for better boundary detection
- Weighted combination of multiple loss functions
- Optimized for rare mineral occurrence detection

## File Descriptions

### Core Model Files

#### `GeoAttNet_model.py`
Defines the DeepUNet architecture with:
- Encoder-decoder structure with skip connections
- Optional CBAM attention modules
- Configurable dropout and channel dimensions
- Input validation for 32×32×15 patches

#### `data_selection.py`
Handles geospatial data processing:
- Multi-raster alignment and stacking
- Label rasterization with buffer zones
- Patch extraction with configurable stride
- Coordinate reference system management

#### `train_GeoAttNet.py`
Training pipeline including:
- Data loading and augmentation
- Loss function implementation (Focal + Dice)
- Learning rate scheduling
- Model checkpointing and validation

#### `test_model.py`
Comprehensive testing and prediction:
- Model evaluation metrics
- Probability map generation
- Confidence estimation
- Visualization and result export

#### `standardize_for_prediction.py`
Data standardization utilities:
- Feature scaling and normalization
- Statistical preprocessing
- Prediction-ready data formatting

### Missing Value Processing

#### `missing_value/geochemical_interpolation.py`
Advanced interpolation algorithms:
- Spatial pattern analysis
- Multiple interpolation methods (RBF, ML, Linear)
- Adaptive method selection
- Quality assessment and validation

#### `missing_value/missing_value_analysis.py`
Missing data pattern analysis:
- Statistical summary of missing values
- Spatial clustering analysis
- Interpolation method recommendations
- Data quality reporting

#### `missing_value/run_interpolation.py`
Automated interpolation pipeline:
- Batch processing of multiple datasets
- Method selection based on missing patterns
- Result validation and export
- Integration with main workflow

## Data Format

### Input Requirements
- **Geophysical/Geochemical Rasters**: Multi-band GeoTIFF files with consistent CRS
- **Mineral Occurrences**: Point data in GeoPackage (.gpkg) format
- **Coordinate System**: EPSG:4326 (WGS84) recommended
- **Resolution**: 0.001° 

### Supported Data Types
- Geochemical element concentrations
- Geophysical measurements (magnetic, gravity)
- Remote sensing derivatives
- Digital elevation models

## Model Architecture

The GeoAttNet model is based on DeepUNet and processes 15-channel geophysical/geochemical patches (32×32 pixels) through:

1. **Encoder Path**: Progressive downsampling with attention
2. **Bottleneck**: Feature compression and representation
3. **Decoder Path**: Upsampling with skip connections
4. **Attention Modules**: CBAM for enhanced feature selection
5. **Output Layer**: Sigmoid activation for probability prediction

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Rasterio 1.2+
- GeoPandas 0.10+
- NumPy 1.21+
- Pandas 1.3+
- Scikit-learn 1.0+
- Matplotlib 3.5+
- Seaborn 0.11+
- TQDM


## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you find bugs or have suggestions.

## Acknowledgments

This work builds upon:
- DeepUNet architecture for semantic segmentation
- CBAM attention mechanisms
- Focal and Dice loss functions for imbalanced data
- Geospatial data processing libraries (Rasterio, GeoPandas)

Thanks to all contributors and open-source projects that made this work possible.