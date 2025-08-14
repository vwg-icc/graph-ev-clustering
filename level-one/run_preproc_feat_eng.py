import time
import glob
import os
import pandas as pd
import joblib
import random
import logging
from tqdm import tqdm
from constants import *
from preprocessing import preprocess
from feature_engineering import feature_engr_sync, feature_engr_parallel
import multiprocessing as mp
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f'processing_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )
    
    return logging.getLogger(__name__)

def apply_parallel_gen(dfGrouped, func, logger):
    """Apply function in parallel to grouped dataframe with progress bar"""
    groups_list = list(dfGrouped)
    
    # Simple parallel execution without nested lambda
    with tqdm(total=len(groups_list), desc="Preprocessing VINs", unit="VIN") as pbar:
        retLst = []
        
        # Process in batches to update progress bar
        batch_size = max(1, len(groups_list) // 20)  # Update progress 20 times
        for i in range(0, len(groups_list), batch_size):
            batch = groups_list[i:i + batch_size]
            batch_results = Parallel(n_jobs=mp.cpu_count())(
                delayed(func)(group[1]) for group in batch
            )
            retLst.extend(batch_results)
            pbar.update(len(batch))
    
    successful_vins = sum(retLst) if retLst else 0
    failed_vins = len(retLst) - successful_vins
    
    logger.info(f"Preprocessing: {len(retLst)} total, {failed_vins} failed")

def apply_parallel_iter(value_list, func, logger, desc="Processing"):
    """Apply function in parallel to value list with progress bar"""
    
    # Simple parallel execution without nested lambda
    with tqdm(total=len(value_list), desc=desc, unit="file") as pbar:
        retLst = []
        
        # Process in batches to update progress bar
        batch_size = max(1, len(value_list) // 20)  # Update progress 20 times
        for i in range(0, len(value_list), batch_size):
            batch = value_list[i:i + batch_size]
            batch_results = Parallel(n_jobs=mp.cpu_count())(
                delayed(func)(val) for val in batch
            )
            retLst.extend(batch_results)
            pbar.update(len(batch))
    
    successful_items = sum(retLst) if retLst else 0
    failed_items = len(retLst) - successful_items
    
    logger.info(f"{desc}: {len(retLst)} total, {failed_items} failed")

if __name__ == "__main__":
    logger = setup_logging()
    print ("Logger initialized. Check ./logs for details.")

    if not IS_PARALLEL:
        logger.info("PARALLEL PROCESSING IS DISABLED - Set IS_PARALLEL = True in constants.py to speed up processing")

    OG_start_time = start_time = time.time()
    folder_path_preprocess = os.path.join(STORAGE_DIR, SAMPLING_RATE)
    folder_path_features = os.path.join(STORAGE_DIR, FREQUENCY)

    if not os.path.exists(folder_path_preprocess): 
        os.makedirs(folder_path_preprocess)
    
    if not os.path.exists(folder_path_features):
        os.makedirs(folder_path_features)

    logger.info("Loading dataset...")
    df = pd.read_parquet(DATA_FILENAME, engine='fastparquet')
    logger.info(f"Dataset loaded - Shape: {df.shape}")
    
    # Select top VINs by frequency
    logger.info(f"Selecting top {TOTAL_VINS} VINs...")
    selected_vins = df.groupby('VIN').size().sort_values(ascending=False).reset_index()["VIN"].tolist()[0:TOTAL_VINS]
    df = df[df['VIN'].isin(selected_vins)]

    grouped = df.groupby('VIN')  # groupby VIN
    
    # Preprocessing phase
    if IS_PARALLEL:
        logger.info("Running preprocessing in parallel mode...")
        apply_parallel_gen(grouped, preprocess, logger)
    else:
        logger.info("Running preprocessing in sequential mode...")
        group_list = list(grouped)
        for i, (vin, group) in enumerate(tqdm(group_list, desc="Preprocessing VINs", unit="VIN")):
            preprocess(group)

    preprocessing_time = (time.time() - start_time) / 60.0
    logger.info(f"Preprocessing completed: {preprocessing_time:.2f} minutes")
    
    # Feature Engineering phase
    start_time = time.time()
    preprocessed_csvfiles = sorted(glob.glob(os.path.join(folder_path_preprocess, '*.csv')))

    if IS_PARALLEL:
        logger.info("Running feature engineering in parallel mode...")
        apply_parallel_iter(preprocessed_csvfiles, feature_engr_parallel, logger, "Feature Engineering")
        
        feature_eng_time = (time.time() - start_time) / 60.0
        logger.info(f"Feature engineering completed in {feature_eng_time:.2f} minutes")
        
        # Scaler fitting phase
        start_time = time.time()
        logger.info("Fitting MinMaxScaler...")
        
        scaler = MinMaxScaler()
        feature_files = sorted(glob.glob(os.path.join(folder_path_features, '*.csv')))
        
        for file_name in tqdm(feature_files, desc="Fitting scaler", unit="file"):
            feat = pd.read_csv(file_name)
            scaler.partial_fit(feat.loc[:, FEATURE_NAMES])
            
        scaler_time = (time.time() - start_time) / 60.0
        logger.info(f"Scaler fitting completed in {scaler_time:.2f} minutes")

    else:
        logger.info("Running feature engineering in sequential mode...")
        
        scaler = MinMaxScaler()
        for _file in tqdm(preprocessed_csvfiles, desc="Feature Engineering", unit="file"):
            f = feature_engr_sync(_file, scaler)
            
        feature_eng_time = (time.time() - start_time) / 60.0
        logger.info(f"Feature engineering completed in {feature_eng_time:.2f} minutes")

    # Save scaler
    logger.info("Saving scaler...")
    scaler_file = os.path.join(folder_path_features, 'scaler.save')
    joblib.dump(scaler, scaler_file)

    # Final summary
    total_time = (time.time() - OG_start_time) / 60.0
    print(f"Processing completed! Total time: {total_time:.2f} minutes, VINs: {len(selected_vins)}")