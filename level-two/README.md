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

---

For questions about Level-Two specifically, refer to the main repository README or contact the development team.
