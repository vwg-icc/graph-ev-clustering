import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.manifold import MDS
import time
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from constants import DIST_FILENAMES, STORAGE_DIR, FREQUENCY,FEATURE_NAMES, WEEK_MEASUREMENTS,FEAT_IDX,FEAT_LABEL, TOTAL_VINS
import time
import seaborn as sns
from utils import get_train_vins

def plot_kde(params):
    """
    Function to plot KDE for various features based on clustering results.
    This function reads the clustering results and the dataset, then generates
    KDE plots for Depth of Discharge, Charging Energy per Cycle, Charging Duration per Cycle,
    and other relevant features.
    """
    
    # Set color palette for clusters
    color_palette = ['#332288',  '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255']

    freq = FREQUENCY # sampling rate
    week = int(10080/int(freq[:-3])) # number of measurements in a week

    features = ['Home','SOC','delta_soc','weekly_mile','dod','charging_power_level','charging_energy_kwh','weekly_cycle',   'delta_mile', 'delta_energy', 'velocity',]
    num_features = len(features) # number of features
    print('features:', num_features)
    folder = STORAGE_DIR + '/' + freq 
    print('folder: ' + folder)

    print('loading data - reading from output')
    #read from saved_csv
    k=params.n_clusters

    cluster_type = params.cluster_algo
    distance=params.distance_metric
    output = pd.read_csv(folder + '/'+cluster_type+'/K_'+str(k)+'_distance_'+distance+'_random_0/output.csv') 

    labels = output['cluster'][output['cluster']>-1].values

    #get the data
    print('loading data - reading from dataset')
    csvfiles = sorted(glob.glob(os.path.join(folder, '*.csv'))) # get list of all files from dataset
    all_df = [] # contains all VIN data
    for file in csvfiles:
        f = pd.read_csv(file)
        all_df.append(f)
    all_df = pd.concat(all_df)

    grouped_df = all_df.groupby('VIN')

    # #depth of discharge
    print('Plotting Depth of Discharge')
    plt.figure()
    for l in sorted(set(labels)):
        cl = output[output['cluster'] == l].reset_index()
        profile = []
        for i in range(len(cl)):
            idx = cl.loc[i,'week']
            if cl.loc[i,'vin'].replace("_", "/") not in grouped_df.groups.keys():
                continue
            cycles = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'cycle_num']
            depth = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'dod']
            data = pd.DataFrame([cycles, depth], index=['cycles','depth']).T.groupby('cycles').max() # depth of discharge per cycle
            data = data[data['depth'] != 0]
            profile.append(data.values)
        profile = np.concatenate(profile).flatten()
        sns.kdeplot(profile, color = color_palette[l], label = f"Cluster {l}")
    plt.title('Depth of Discharge per Cycle')
    plt.xlabel('Depth of Discharge [%]', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.grid(axis='y', alpha=0.4)
    plt.legend()
    plt.savefig('Figures/KDE_Kmeans_DOD_per_cycle_'+cluster_type+'_KDE.pdf')


    # charging energy per cycle
    print('Plotting Charging Energy per cycle')
    plt.figure()
    for l in sorted(set(labels)):
        cl = output[output['cluster'] == l].reset_index()
        profile = []
        for i in range(len(cl)):
            idx = cl.loc[i,'week']
            if cl.loc[i,'vin'].replace("_", "/") not in grouped_df.groups.keys():
                continue
            cycles = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'cycle_num']
            energy = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'charging_energy_kwh']
            data = pd.DataFrame([cycles, energy], index=['cycles','energy']).T.groupby('cycles').max() # sum of charging energy per cycle
            data = data[data['energy'] != 0]
            profile.append(data.values)
        profile = np.concatenate(profile).flatten()
        sns.kdeplot(profile, color = color_palette[l], label = f"Cluster {l}")
    plt.title('Charging Energy per Cycle')
    plt.xlabel('Charging Energy [kWh]', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.legend()
    plt.grid(axis='y', alpha=0.4)
    plt.savefig('Figures/Charging_energy_per_cycle_'+cluster_type+'_KDE.pdf')
    plt.show()

    # total charging energy per week
    print('Plotting Total Charging Energy per Week')
    plt.figure()
    for l in sorted(set(labels)):
        cl = output[output['cluster'] == l].reset_index()
        profile = []
        for i in range(len(cl)):
            
            idx = cl.loc[i,'week']
            if cl.loc[i,'vin'].replace("_", "/") not in grouped_df.groups.keys():
                continue
            cycles = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'cycle_num']
            energy = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'charging_energy_kwh']
            data = pd.DataFrame([cycles, energy], index=['cycles','energy']).T.groupby('cycles').max() # charging energy per cycle
            data = sum(data['energy'])
            profile.append(data)
        sns.kdeplot(profile,  color = color_palette[l], label = f"Cluster {l}")
    plt.title('Charging Energy per Week')
    plt.xlabel('Charging Energy per Week [kWh]', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.grid(axis='y', alpha=0.4)
    plt.legend()
    plt.savefig('Figures/Charging_energy_per_week_'+cluster_type+'_KDE.pdf')
    plt.show()

    # charging duration per cycle
    print('Plotting Charging Duration per Cycle')
    plt.figure()
    for l in sorted(set(labels)):
        cl = output[output['cluster'] == l].reset_index()
        profile = []
        for i in range(len(cl)):
            idx = cl.loc[i,'week']
            if cl.loc[i,'vin'].replace("_", "/") not in grouped_df.groups.keys():
                continue
            cycles = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'cycle_num']
            time = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'charging_status']
            data = pd.DataFrame([cycles, time], index=['cycles','time']).T.groupby('cycles').sum()
            data = data[data['time']>0]*int(freq[:-3])/60
            profile.append(data['time'].values)
        profile = np.concatenate(profile)
        sns.kdeplot(profile,  color = color_palette[l], label = f"Cluster {l}")
    plt.title('Charging Duration per Cycle')
    plt.xlabel('Charging Time per Cycle [hr]', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.grid(axis='y', alpha=0.4)
    plt.legend()
    plt.savefig('Figures/charging_duration_per_cycle_'+cluster_type+'_KDE.pdf')
    plt.show()

    # total charging time per week
    print('Total Charging Time per Week')
    plt.figure()
    for l in sorted(set(labels)): #labels = [0,1,2,3,4,5]
        cl = output[output['cluster'] == l].reset_index() #output is a table with VIN, week (a number), cluster (a number)
        profile = []
        for i in range(len(cl)): #i goes over all VIN weeks in that cluster
            idx = cl.loc[i,'week'] # idx is basically week
            if cl.loc[i,'vin'].replace("_", "/") not in grouped_df.groups.keys():
                continue
            #from original data get VIN data, then get a specific week from week0:week1 (1000 measurements) and extract charging_status for all
            data = grouped_df.get_group(cl.loc[i,'vin'].replace("_", "/")).loc[week*idx:week*(idx+1),'charging_status'].sum()
            profile.append(data*int(freq[:-3])/60)
        sns.kdeplot(profile, color = color_palette[l], label = f"Cluster {l}")
    plt.title('Charging Duration per Week')
    plt.xlabel('Charging Time [hr]', fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.grid(axis='y', alpha=0.4)
    plt.legend()
    plt.savefig('Figures/charging_duration_per_week_'+cluster_type+'_KDE.pdf')
    plt.show()