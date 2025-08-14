# Level-Two: Graph-Based Community Detection

This directory contains the second level of the EV clustering pipeline, which uses advanced graph neural networks to identify higher-level community structures from Level-One clustering results.

## Overview

Level-Two takes the clustering results from Level-One and applies an End-to-End (ETE) cluster model using PyTorch and PyTorch Geometric to discover community patterns among vehicles that exhibit similar long-term usage behaviors.

## Key Features

- **Graph Neural Networks**: Uses LSTM encoders combined with Graph Convolutional Networks (GCN)
- **DMoN Pooling**: Deep Modularity Networks for effective clustering
- **Multi-loss Training**: Combines spectral, orthogonality, and cluster losses
- **Comprehensive Visualization**: Generates heatmaps, histograms, and community statistics
- **Configurable Architecture**: Easy parameter tuning through configuration files

## Prerequisites

- Level-One clustering results (CSV files with cluster assignments)
- Python environment with PyTorch and PyTorch Geometric installed
- Configured parameters in `config.py`

## Configuration

Edit `config.py` to customize the following sections:

### Data Configuration (`DATA_CONFIG`)
```python
DATA_CONFIG = {
    'csv_file': '../level-one/files/preprocessed/VINS_50/10min/kmedoids/K_6_distance_manhattan_random_0/output.csv',
    'method': 'kmedoids',  # clustering method used in level-one
    'index_column': 'vin',
    'columns': 'week',
    'values': 'cluster',
    'fill_na_value': -2,
    'drop_columns': ['covid']  # columns to exclude from output
}
```

### Model Configuration (`MODEL_CONFIG`)
```python
MODEL_CONFIG = {
    'dim': 1,                    # input feature dimension
    'encoder_size': 128,         # LSTM encoder hidden size
    'num_neighbors': 15,         # number of neighbors for graph construction
    'num_clusters': 12,          # target number of communities
    'random_seed': 11            # for reproducibility
}
```

### Training Configuration (`TRAINING_CONFIG`)
```python
TRAINING_CONFIG = {
    'max_epochs': 75,           # maximum training epochs
    'learning_rate': 0.01       # Adam optimizer learning rate
}
```

### Visualization Configuration (`VIZ_CONFIG`)
```python
VIZ_CONFIG = {
    'figure_size_heatmap': (10, 10),      # heatmap dimensions
    'figure_size_histogram': (12, 8),     # histogram dimensions
    'weeks_to_display': 52,               # number of weeks to show
    'output_format': 'pdf',               # output format (pdf/png)
    'dpi': 150                            # image resolution
}
```

## Usage

### Basic Usage

1. **Configure the model**: Edit `config.py` with your desired parameters
2. **Run training**: Execute the main training script

```bash
cd level-two
python train_model.py
```

### What the Script Does

The `train_model.py` script performs the following operations:

1. **Data Loading**: Loads Level-One clustering results from CSV
2. **Data Preparation**: Converts to tensor format and handles missing values
3. **Model Training**: Trains the ETE cluster model with configured parameters
4. **Community Detection**: Generates final community assignments
5. **Visualization**: Creates comprehensive plots and saves results
6. **Output Generation**: Saves community assignments to CSV files

### Training Process

The model combines three loss functions:
- **Spectral Loss**: Encourages proper graph structure
- **Orthogonality Loss**: Ensures distinct community representations
- **Cluster Loss**: Promotes well-separated clusters

Training progress is logged every 10 epochs showing individual and total losses.

## Outputs

### Community Assignment Files
- CSV files with community labels for each vehicle
- Filename pattern: `community_{method}_k_6_random_seed_{seed}_max_{clusters}.csv`

### Visualizations
All visualizations are saved in the `figures/` directory:

#### Heatmaps (`figures/heatmaps/`)
- Show cluster patterns within each community over time
- One heatmap per detected community
- Color-coded cluster assignments with legend

#### Histograms (`figures/histograms/`)
- Display distribution of Level-One clusters within each community
- Frequency analysis of cluster labels per community
- Helps understand community composition

#### Community Statistics (`figures/community_counts/`)
- Bar chart showing number of vehicles per community
- Overall distribution of community sizes
- Useful for understanding community balance

## Model Architecture

The ETE Cluster Model consists of:

1. **LSTM Encoder**: Processes time series data to create embeddings
2. **Graph Construction**: Builds k-nearest neighbor graphs from embeddings
3. **Graph Convolution**: Applies ClusterGCNConv for node representation learning
4. **DMoN Pooling**: Performs differentiable clustering with modularity optimization
5. **Multi-objective Loss**: Combines multiple objectives for robust training

## Key Components

- `train_model.py`: Main training script with full pipeline
- `ETEClusterModel.py`: Neural network model definition
- `config.py`: Configuration parameters for all components
- `Autoencoder.py`: Additional neural network utilities

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The model automatically detects GPU availability
2. **Memory Errors**: Reduce batch size or model dimensions in `MODEL_CONFIG`
3. **Convergence Issues**: Adjust learning rate or increase epochs in `TRAINING_CONFIG`
4. **File Path Errors**: Ensure Level-One outputs exist at specified paths

### Configuration Tips

- Start with default parameters and adjust incrementally
- Use `random_seed` for reproducible results
- Monitor loss values to ensure proper convergence
- Adjust `num_clusters` based on your dataset size and requirements

## Dependencies

Key requirements:
- PyTorch
- PyTorch Geometric  
- NumPy
- Pandas
- Matplotlib
- Seaborn

See the main `requirements.txt` for complete dependency list.

---

For questions about Level-Two specifically, refer to the main repository README or contact the development team.
