# Level-One: Data Preprocessing and Clustering Pipeline

This directory contains the first level of the data processing pipeline, which handles data preprocessing, feature engineering, and level-one clustering analysis for vehicle time-series data.

## Overview

The Level-One pipeline consists of two main stages:

1. **Data Preprocessing & Feature Engineering** (`run_preproc_feat_eng.py`)
2. **Clustering Experiment** (`run_experiment.py`)

The output from these steps is stored in the structured directory path and serves as input for level-two clustering.

## Main Scripts

#### 1. `run_preproc_feat_eng.py`

**Purpose**: Preprocesses raw vehicle data and performs feature engineering to prepare data for clustering analysis.

**Key Functionality**:
- **Data Loading**: Loads vehicle time-series data from parquet files
- **VIN Selection**: Selects top N vehicles by data frequency (configurable via `TOTAL_VINS`)
- **Preprocessing**: Cleans and processes raw time-series data per vehicle
- **Feature Engineering**: Extracts meaningful features from time-series data
- **Parallel Processing**: Supports both parallel and sequential processing modes
- **Data Scaling**: Fits and saves MinMaxScaler for feature normalization

**Processing Pipeline**:
1. Load raw dataset from `DATA_FILENAME`
2. Select top `TOTAL_VINS` vehicles by data frequency
3. Group data by VIN and preprocess each vehicle's time-series
4. Extract engineered features from preprocessed data
5. Fit MinMaxScaler on all feature data
6. Save preprocessed files and scaler

**Output Structure**:
- Preprocessed data: `files/preprocessed/VINS_50/1min/*.csv` (per vehicle)
- Feature data: `files/preprocessed/VINS_50/10min/*.csv` (per vehicle)
- Scaler: `files/preprocessed/VINS_50/10min/scaler.save`

**Performance**:
- Supports parallel processing using all CPU cores
- Progress tracking with tqdm progress bars
- Comprehensive logging to `./logs/` directory

#### 2. `run_experiment.py`

**Purpose**: Performs clustering analysis on the engineered features and generates final output for Level-Two processing.

**Key Functionality**:
- **Configuration Management**: Loads clustering parameters from `config.json`
- **Data Preparation**: Creates feature arrays from preprocessed data
- **Clustering**: Executes clustering algorithm (K-medoids, K-means, etc.)
- **Visualization**: Generates cluster plots and load profile characterizations
- **Output Generation**: Creates structured output CSV for Level-Two analysis

**Processing Pipeline**:
1. Load configuration from `config.json`
2. Create data arrays from feature files
3. Execute clustering algorithm with specified parameters
4. Generate cluster visualizations and plots
5. Create comprehensive output dataframe with cluster assignments
6. Save results to structured output directory

**Output Location**:
```
files/preprocessed/VINS_50/10min/kmedoids/K_6_distance_manhattan_random_0/output.csv
```

**Output Structure**: The output CSV contains:
- Vehicle identifiers (VINs)
- Time period information (weeks)
- Cluster assignments
- Feature values and metadata


#### 3. `Error Bar Analysis (Optional - for optimal k-value selection)`:
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

## Configuration

### `config.json`
Controls clustering experiment parameters:

```json
{
    "features_dir": "files/preprocessed/VINS_50/10min/",
    "distance_threshold": 500,
    "n_clusters": 6,
    "cluster_algo": "kmedoids",
    "max_iter": 300,
    "random_state": 0,
    "distance_metric": "manhattan",
    "precomputed": "precomputed",
    "experiment": "1",
    "dr": false
}
```

### `constants.py`
Defines system-wide constants:
- **Sampling rates**: `SAMPLING_RATE = '1min'`, `FREQUENCY = '10min'`
- **Dataset configuration**: `TOTAL_VINS = 50`, data paths
- **Feature definitions**: 11 engineered features including SOC, mileage, charging patterns
- **Processing options**: `IS_PARALLEL = True`

## Features Extracted

The pipeline extracts 11 key features for clustering:

| Feature | Description |
|---------|-------------|
| `Home` | Home location indicator |
| `SOC` | State of Charge percentage |
| `delta_soc` | Change in State of Charge |
| `weekly_mile` | Weekly mileage |
| `dod` | Depth of Discharge percentage |
| `charging_power_level` | Charging power level |
| `charging_energy_kwh` | Charging energy in kWh |
| `weekly_cycle` | Weekly cycle number |
| `delta_mile` | Change in mileage |
| `delta_energy` | Change in energy |
| `velocity` | Vehicle velocity |

## Usage

### Step 1: Data Preprocessing and Feature Engineering
```bash
cd level-one
python run_preproc_feat_eng.py
```

This will:
- Process raw vehicle data
- Generate engineered features
- Save preprocessed files and scaler
- Log progress to `./logs/`

### Step 2: Clustering Experiment
```bash
python run_experiment.py
```

This will:
- Load configuration from `config.json`
- Perform clustering analysis
- Generate visualizations
- Save output CSV for Level-Two processing

## Output for Level-Two

The final output is stored at:
```
files/preprocessed/VINS_50/10min/kmedoids/K_6_distance_manhattan_random_0/output.csv
```

This CSV file contains:
- **Cluster assignments** for each vehicle-week combination
- **Feature vectors** used in clustering
- **Metadata** about vehicles and time periods
- **Quality metrics** and clustering results

The output directory structure follows the pattern:
```
files/preprocessed/VINS_{N}/{frequency}/{algorithm}/K_{clusters}_distance_{metric}_random_{seed}/
```


