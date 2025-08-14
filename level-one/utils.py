import os
import glob
import numpy as np
import pandas as pd
import joblib
import umap.umap_ as umap
from sklearn.manifold import MDS
import time
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from tslearn.barycenters import euclidean_barycenter
from matplotlib.backends.backend_pdf import PdfPages
from constants import DIST_FILENAMES, STORAGE_DIR, FREQUENCY,FEATURE_NAMES, WEEK_MEASUREMENTS,FEAT_IDX,FEAT_LABEL, TOTAL_VINS

def get_distance_matrix(data, distance_metric, directory):
    """
    Computes the pairwise Euclidean distance matrix between data points.

    Args:
        data (numpy.ndarray): A 2D array of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: A 2D array of shape (n_samples, n_samples) containing
            the pairwise Euclidean distances between data points.
    """
    # filename = f"{STORAGE_DIR}/{FREQUENCY}/VINS_{TOTAL_VINS}/{DIST_FILENAMES[distance_metric]}"
    filename = f"{directory}{DIST_FILENAMES[distance_metric]}"
    print (f"Checking for distance matrix file... {filename}")
    if not os.path.exists(filename):
        print (f"File not found, recalculating...")
        if distance_metric == "manhattan":
            distance_matrix = cdist(data, data, metric='cityblock')
        
        elif distance_metric == "euclidean":
            distance_matrix = cdist(data, data, metric='euclidean')

        elif distance_metric == "minkowski":
            distance_matrix = cdist(data, data, metric='minkowski', p=4)
        
        elif distance_metric == "cosine":
            distance_matrix = cdist(data, data, 'cosine')
            
        # save distance matrix to file
        np.save(filename, distance_matrix)

    else:
        print(f"Path for distance matrix exists..loading existing..")
        distance_matrix = np.load(filename)
        
    return distance_matrix


def create_data_array(directory):
    """
    Reads the .csv files in the input list, processes them, and returns a Numpy array.

    Args:
    directory (list): A path to the features stored as VIN names that have FREQUENCY 
    separated data for every VIN.

    Returns:
    np_array (numpy.ndarray): A Numpy array containing the vehicle-week data

    """
    data_raw = []
    valid_vins = []
    drop_vins = []
    valid_weeks = []
    drop_weeks = []
    scaler = joblib.load(f"{directory}scaler.save") # dataset folder
    files = sorted(glob.glob(os.path.join(directory, '*.csv'))) # list of all files from dataset folder
    for vin_file in files: # loop through VIN files
        vin_df = pd.read_csv(vin_file)
        data_features = vin_df.loc[:,FEATURE_NAMES].copy() # extract features
        data_features_norm = scaler.transform(data_features) # normalization
        data_features = data_features.values
        for w in range(0, len(data_features)-WEEK_MEASUREMENTS+1, WEEK_MEASUREMENTS): # sliding window (size = week, stride = week)
            window = data_features[w:w+WEEK_MEASUREMENTS,:]
            if sum(abs(window.T[2])) < 10 or  -1 in window.T[1]: # drop vehicle-weeks where total absolute change in SOC < 10
                backslash = '\\'
                drop_vins.append(vin_file.split(backslash)[-1].split('/')[-1][:-4])
                #drop_vins.append(vin_file.split('/')[-1][:-4])
                drop_weeks.append(vin_df.loc[w,'week_num'])
            else:
                backslash = '\\'
                valid_vins.append(vin_file.split(backslash)[-1].split('/')[-1][:-4])
                #valid_vins.append(vin_file.split('/')[-1][:-4])
                valid_weeks.append(vin_df.loc[w,'week_num'])
                window = data_features_norm[w:w+WEEK_MEASUREMENTS,:]
                data_raw.append(window)
    return np.array(data_raw), valid_vins, drop_vins, valid_weeks, drop_weeks, scaler

def create_non_drop_data_arr(directory):
    """
        Returns a numpy array containing all vehicle-weeks (no dropping)
    """
    files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    data_raw = []
    for vin_file in files: # loop through VIN files
        vin_df = pd.read_csv(vin_file)
        data_features = vin_df.loc[:,FEATURE_NAMES].copy().values # extract features
        for w in range(0, len(data_features)-WEEK_MEASUREMENTS+1, WEEK_MEASUREMENTS): # sliding window (size = week, stride = week)
            window = data_features[w:w+WEEK_MEASUREMENTS,:]
            data_raw.append(window)
    return np.array(data_raw)

def get_train_vins(directory):
    csvfiles = sorted(glob.glob(os.path.join(directory, '*.csv'))) # list of all files from dataset folder
    train_files = csvfiles
    train_vins = [] # list of VINs in training set
    for file in train_files:
        #train_vins.append(file.split('/')[-1][:-4])
        backslash = '\\'
        train_vins.append(file.split(backslash)[-1].split('/')[-1][:-4])
    num_train_vins = len(train_files) # number of VINs in training set
    print (f"Num of VINS: {num_train_vins}")
    return train_vins


def generate_output_df(train_vins, directory, valid_train_vins, valid_train_weeks, drop_train_vins, drop_train_weeks, labels):
    # create dataframe to store labels
    valid = pd.DataFrame(zip(valid_train_vins, valid_train_weeks, labels), 
                        columns=['vin', 'week', 'cluster'])
    dropped = pd.DataFrame(zip(drop_train_vins, drop_train_weeks, [-1]*len(drop_train_vins)),
                        columns=['vin','week','cluster'])
    output = pd.concat([valid, dropped], ignore_index=True)
    output = output.sort_values(['vin','week']).reset_index(drop=True)
    output['index'] = output.index

    
    # get dates
    print (f"Len of train_vins: {len(train_vins)}")
    for vin in train_vins:
        f = pd.read_csv(f"{directory}{vin}.csv")
        for i in range(len(f)//WEEK_MEASUREMENTS):
            output.loc[(output['vin'] == vin) & (output['week'] == i), 'date'] = f['datetime'][i*WEEK_MEASUREMENTS][:-9]

    # pre-Covid = 0, Covid = 1
    for i in range(len(output)):
        if output['date'][i] > '2020-03-11':
            output.loc[i,'covid'] = 1
        else:
            output.loc[i,'covid'] = 0
    print (f"{output.shape}")
    return output





