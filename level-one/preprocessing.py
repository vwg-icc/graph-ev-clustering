import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from constants import LAST_DATE_CONST, SAMPLING_RATE, STORAGE_DIR, FREQUENCY, LIMIT_INTERP

def preprocess(df: pd.DataFrame) -> int:
    """
    Preprocess time series data with datetime indexing, resampling, and interpolation.
    
    Args:
        df: DataFrame containing columns: Timestamp, VIN, Home, SOC, Mileage
        
    Returns:
        int: 1 if successful, 0 if dataframe is empty after processing
    """
    if df.empty:
        return 0
    
    # Validate required columns
    required_columns = ['Timestamp', 'VIN', 'Home', 'SOC', 'Mileage']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    try:
        # Create a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Set up datetime index
        df_processed = _setup_datetime_index(df_processed)
        
        # Resample and interpolate data
        df_processed = _resample_and_interpolate(df_processed)
        
        # Handle special mileage interpolation
        df_processed = _handle_mileage_gaps(df_processed)
        
        # Add temporal features and filter data
        df_processed = _add_temporal_features_and_filter(df_processed)
        
        # Save processed data
        return _save_processed_data(df_processed)
        
    except Exception as e:
        raise

def _setup_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set up datetime index and remove duplicates."""
    df.index = pd.to_datetime(df['Timestamp'])
    df.drop(columns=['Timestamp'], inplace=True)
    
    # Remove duplicates (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    return df

def _resample_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Resample categorical and numerical data with appropriate methods."""
    original_index = df.index
    new_index = pd.date_range(original_index.min(), original_index.max(), freq=SAMPLING_RATE)
    
    # Handle categorical features with forward/backward fill
    categorical_cols = ['VIN', 'Home']
    df_categorical = df[categorical_cols].resample(SAMPLING_RATE).ffill().bfill()
    df_categorical.index = df_categorical.index.floor('Min')
    
    # Handle numerical features with interpolation
    numerical_cols = ['SOC', 'Mileage']
    df_numerical = (df[numerical_cols]
                   .reindex(original_index.union(new_index))
                   .interpolate('index', limit=LIMIT_INTERP)
                   .reindex(new_index))
    
    df_numerical = df_numerical.fillna(-1)
    df_numerical.index = df_numerical.index.floor('Min')
    
    # Combine results
    return pd.concat([df_categorical, df_numerical], axis=1)

def _handle_mileage_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Handle special case interpolation for mileage gaps."""
    # Reset index temporarily for easier processing
    df_temp = df.copy()
    df_temp["Timestamp"] = df_temp.index
    df_temp.index = range(len(df_temp))
    
    # Find indices where mileage is -1 (missing)
    missing_indices = df_temp.index[df_temp['Mileage'] == -1].tolist()
    
    if not missing_indices:
        df_temp.set_index("Timestamp", inplace=True)
        return df_temp
    
    # Find consecutive gaps and their boundaries
    gap_ranges = _find_consecutive_gaps(missing_indices, len(df_temp))
    
    # Apply special interpolation for small mileage differences
    for start_idx, end_idx in gap_ranges.items():
        if start_idx >= 0 and end_idx < len(df_temp):
            mileage_diff = abs(df_temp.loc[start_idx, "Mileage"] - 
                             df_temp.loc[end_idx, "Mileage"])
            
            if mileage_diff < 1:  # Small difference threshold
                df_temp.loc[start_idx + 1:end_idx - 1, ["Mileage", "SOC"]] = np.nan
    
    # Final interpolation - only interpolate numerical columns
    numerical_cols = ['SOC', 'Mileage']
    df_temp[numerical_cols] = df_temp[numerical_cols].interpolate('linear')
    df_temp.set_index("Timestamp", inplace=True)
    
    return df_temp

def _find_consecutive_gaps(missing_indices: list, df_length: int) -> dict:
    """Find consecutive gaps in missing indices and return their boundaries."""
    if not missing_indices:
        return {}
    
    gap_ranges = {}
    start_idx = missing_indices[0]
    
    for i in range(len(missing_indices) - 1):
        current_idx = missing_indices[i]
        next_idx = missing_indices[i + 1]
        
        # If not consecutive or at the end
        if next_idx - current_idx > 1 or i + 1 == len(missing_indices) - 1:
            if i + 1 == len(missing_indices) - 1 and next_idx - current_idx == 1:
                # Handle last consecutive pair
                end_idx = min(next_idx + 2, df_length - 1)
            else:
                end_idx = current_idx + 1
            
            if start_idx > 0 and end_idx < df_length:
                gap_ranges[start_idx - 1] = end_idx
            
            # Start new gap range
            if next_idx - current_idx > 1:
                start_idx = next_idx
    
    return gap_ranges

def _add_temporal_features_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features and apply date filters."""
    # Add datetime column
    df.insert(0, 'datetime', df.index)
    
    # Find first Monday to start the time series
    df['weekday'] = df['datetime'].dt.day_of_week
    first_monday_idx = (df['weekday'] == 0).idxmax() if (df['weekday'] == 0).any() else 0
    
    if isinstance(first_monday_idx, bool):  # No Monday found
        first_monday_idx = 0
    else:
        first_monday_idx = df.index.get_loc(first_monday_idx)
    
    # Filter data: start from first Monday and end before LAST_DATE_CONST
    df = df.iloc[first_monday_idx:]
    df = df[df['datetime'] < LAST_DATE_CONST]
    
    # Clean up
    df.drop(columns=['weekday'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def _save_processed_data(df: pd.DataFrame) -> int:
    """Save processed dataframe to CSV file."""
    if df.empty:
        return 0
    
    # Create output directory if it doesn't exist
    output_dir = Path(STORAGE_DIR) / SAMPLING_RATE
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate safe filename
    vin = str(df["VIN"].iloc[0]).replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{vin}.csv"
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return 1