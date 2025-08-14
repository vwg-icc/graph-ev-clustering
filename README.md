# EV Usage Clustering with EnergyAI

**Official implementation of our paper**: [Graph-based two-level clustering for electric vehicle usage patterns](https://doi.org/10.1016/j.egyai.2025.100539) (Energy and AI, 2025)

A comprehensive two-level clustering framework for analyzing electric vehicle (EV) charging and usage patterns using time series data and graph neural networks.

## Description

This repository implements a hierarchical clustering approach to analyze EV usage patterns:

- **Level-One**: Traditional time series clustering on preprocessed EV data using various algorithms (K-means, K-medoids, Hierarchical clustering, etc.)
- **Level-Two**: Advanced graph-based clustering using End-to-End (ETE) cluster models with Graph Neural Networks to identify higher-level community structures

The framework processes EV telemetry data including State of Charge (SOC), mileage, charging status, and temporal features to discover meaningful usage patterns and driver behaviors.

## Installation

1. Clone the repository:
```bash
git clone https://gitlab.yaday.io/guard-ion/ev-usage-clustering-energyai.git
cd ev-usage-clustering-energyai
```

2. Create and activate a virtual environment:
```bash
python -m venv ev-clustering-env
source ev-clustering-env/bin/activate  
# On Windows: ev-clustering-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Level-One: Data Preprocessing and Initial Clustering

The Level-One pipeline handles data preprocessing, feature engineering, and initial clustering analysis.

#### Prerequisites
- Raw EV telemetry data in the expected format (Timestamp, VIN, Home, SOC, Mileage)
- Configure parameters in `level-one/config.json`

#### Configuration
Edit `config.json` to customize:
- Data paths and sampling rates
- Clustering algorithms and parameters
- Feature engineering settings
- Output directories

#### Running the Pipeline

1. **Data Preprocessing and Feature Engineering**:
```bash
cd level-one
python run_preproc_feat_eng.py
```
This script:
- Preprocesses raw time series data
- Performs resampling and interpolation
- Engineers relevant features (delta SOC, charging cycles, weekly patterns)
- Outputs processed data for clustering

2. **Clustering Experiment**:
```bash
python run_experiment.py
```
This script:
- Loads preprocessed data
- Runs specified clustering algorithms
- Generates cluster assignments
- Creates visualizations and saves results

3. **Error Bar Analysis** (Optional - for optimal k-value selection):
```bash
python run_error_bar_experiments.py --config config.json \
    --k_clusters 5 6 7 8 9 \
    --random_seed_vals 0 1 2 3 4 5 6 7 8 9 \
    --n_jobs 4 \
    --save_csv True \
    --output_dir ./results/
```
This script:
- Runs clustering experiments across multiple k-values and random seeds
- Generates error bar data for determining optimal cluster numbers
- Outputs results that can be used for "picking the right k-value" analysis
- Creates CSV files with clustering metrics for statistical analysis

#### Key Components
- `preprocessing.py`: Time series preprocessing utilities
- `feature_engineering.py`: Feature extraction and engineering
- `CreateCluster.py`: Clustering algorithm implementations
- `utils.py`: Visualization and analysis utilities

### Level-Two: Graph-Based Community Detection

The Level-Two pipeline uses advanced graph neural networks to identify higher-level community structures from Level-One clustering results.

#### Prerequisites
- Level-One clustering results (CSV files with cluster assignments)
- Configure parameters in `level-two/config.py`

#### Running the Pipeline

1. **Configure the model**:
Edit `config.py` to set:
- Data paths pointing to Level-One outputs
- Model architecture parameters
- Training configuration
- Visualization settings

2. **Train the ETE Cluster Model**:
```bash
cd level-two
python train_model.py
```

This script:
- Loads Level-One clustering results
- Trains an End-to-End cluster model using PyTorch and PyTorch Geometric
- Generates community assignments
- Creates heatmaps and histogram visualizations
- Saves community assignments to CSV

#### Key Components
- `ETEClusterModel.py`: Graph neural network architecture for clustering
- `train_model.py`: Main training script with visualization
- `Autoencoder.py`: Neural network components for feature encoding
- `config.py`: Configuration settings for model and training


## Dependencies

Key libraries used:
- **Data Processing**: pandas, numpy, scikit-learn
- **Machine Learning**: tslearn, sklearn-extra, hdbscan
- **Deep Learning**: PyTorch, PyTorch Geometric, TensorFlow/Keras
- **Visualization**: matplotlib, seaborn, plotly
- **Utilities**: joblib, tqdm, umap-learn, networkx

See `requirements.txt` for the complete list.

## Development Status

⚠️ **Note**: Some visualization and analysis components are still work in progress and will be uploaded soon. This includes:
- Advanced plotting utilities in Level-One
- Extended visualization options

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{balaram2025graph,
  title={Graph-based two-level clustering for electric vehicle usage patterns},
  author={Balaram, Dhanashree and Dufford, Brett and Martin, Sonia and Negoita, Gianina Alina and Yen, Matthew and Paxton, William A.},
  journal={Energy and AI},
  pages={100539},
  year={2025},
  publisher={Elsevier},
  issn={2666-5468},
  doi={10.1016/j.egyai.2025.100539},
  url={https://www.sciencedirect.com/science/article/pii/S2666546825000710}
}
```

---

For questions or issues, please open an issue on the repository or contact the development team.
