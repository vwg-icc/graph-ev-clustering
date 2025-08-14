# ───────────────────────────────────────────────────────────────
#   Sampling and Frequency Settings
# ───────────────────────────────────────────────────────────────
SAMPLING_RATE = '1min'       # Raw time-series sampling rate (before downsampling)
FREQUENCY = '10min'          # Desired resampling interval (used in analysis)
DOWNSAMPLE_VALUE = int(FREQUENCY[:-3])  # Downsampling ratio (e.g., 10 for "10min")
WEEK_MEASUREMENTS = int(10080 / DOWNSAMPLE_VALUE)  # 10080 = minutes in a week

# ───────────────────────────────────────────────────────────────
#   Energy & Domain Constants
# ───────────────────────────────────────────────────────────────
THERMAL_ENERGY_CONST = 83.6  # Used for domain-specific energy calculations
LIMIT_INTERP = 1440          # Limit for interpolation (e.g., 1 day worth of minutes)
LAST_DATE_CONST = '2020-06-01 00:00:00'  # Last timestamp considered in dataset

# ───────────────────────────────────────────────────────────────
#   File Paths and Dataset Configuration
# ───────────────────────────────────────────────────────────────
DATA_FILENAME = "data/car-demo-data.parquet"  # Raw dataset location
TOTAL_VINS = 3200                                     # Number of vehicles (VINs) to use
STORAGE_DIR = f'files/preprocessed/VINS_{TOTAL_VINS}'  # Output directory

# ───────────────────────────────────────────────────────────────
#   Distance Matrix File Mapping
# ───────────────────────────────────────────────────────────────
DIST_FILENAMES = {
    "euclidean":  "train_euc_dist.npy",
    "manhattan":  "train_man_dist.npy",
    "minkowski":  "train_mink_dist.npy",
    "cosine":     "train_cos_dist.npy",
    "dtw":        "train_dtw_dist.npy"
}

# ───────────────────────────────────────────────────────────────
#   Processing Options
# ───────────────────────────────────────────────────────────────
IS_PARALLEL = True  # Toggle multiprocessing or parallel execution

# ───────────────────────────────────────────────────────────────
#   Feature Configuration
# ───────────────────────────────────────────────────────────────
# List of features selected for clustering
FEATURE_NAMES = [
    'Home', 'SOC', 'delta_soc', 'weekly_mile', 'dod',
    'charging_power_level', 'charging_energy_kwh', 'weekly_cycle',
    'delta_mile', 'delta_energy', 'velocity'
]

# Index mapping for fast access into feature vectors
FEAT_IDX = {
    'Home': 0,
    'SOC': 1,
    'delta_soc': 2,
    'weekly_mile': 3,
    'dod': 4,
    'charging_power_level': 5,
    'charging_energy_kwh': 6,
    'weekly_cycle': 7,
    'delta_mile': 8,
    'delta_energy': 9,
    'velocity': 10
}

# Human-readable labels for visualization and interpretation
FEAT_LABEL = {
    'Home': 'Home',
    'SOC': 'SOC%',
    'delta_soc': 'Change in SOC%',
    'weekly_mile': 'Mileage',
    'dod': 'Depth of Discharge%',
    'charging_power_level': 'Charging Power Level',
    'charging_energy_kwh': 'Charging Energy kWh',
    'weekly_cycle': 'Cycle Number',
    'temp': 'Temperature',
    'delta_temp': 'Change in Temperature',
    'delta_mile': 'Change in mileage',
    'delta_energy': 'Change in energy',
    'velocity': 'Velocity'
}

# ───────────────────────────────────────────────────────────────
#  Notes for Feature Engineering
# - Energy usage differences: high delta_SOC / low mileage may imply heating/comfort usage.
# - Penalize range for temperature deviation from comfortable range.
# - Features like temp, delta_temp can be considered for further refinement.
# ───────────────────────────────────────────────────────────────
