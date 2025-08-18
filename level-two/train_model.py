"""
ETE Cluster Model Training Script

This script loads time series data, trains an ETE cluster model,
generates visualizations of the resulting clusters, and saves community assignments.
"""

import logging
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
import os

from ETEClusterModel import ETEClusterModel
from config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIZ_CONFIG, OUTPUT_CONFIG

# Configure logger to output to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_output_directories():
    """
    Create the necessary output directories if they don't exist.
    """
    directories = [
        'figures',
        'figures/heatmaps',
        'figures/histograms',
        'figures/community_counts'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")


def load_and_prepare_data():
    """
    Load data from CSV and prepare it for training.
    
    Returns:
        tuple: (X tensor, original dataframe)
    """
    logger.info("Loading data...")
    csv_file = DATA_CONFIG['csv_file']
    logger.info(f"Using {DATA_CONFIG['method']} data from {csv_file}")
    df = pd.read_csv(csv_file)
    data = df.pivot(
        index=DATA_CONFIG['index_column'], 
        columns=DATA_CONFIG['columns'], 
        values=DATA_CONFIG['values']
    ).dropna(axis=1, how='all')
    logger.info(f"Data shape: {data.shape}")
    
    # Convert to tensor and reshape for model input
    X = torch.tensor(
        data.fillna(DATA_CONFIG['fill_na_value']).values
    ).reshape(data.shape[0], data.shape[1], 1).float()
    
    logger.info(f"Tensor shape: {X.shape}")
    return X, df


def initialize_model():
    """
    Initialize the ETE cluster model with configured parameters.
    
    Returns:
        ETEClusterModel: Initialized model
    """
    logger.info("Initializing model...")
    
    model = ETEClusterModel(
        dim=MODEL_CONFIG['dim'],
        encoder_size=MODEL_CONFIG['encoder_size'],
        num_neighbors=MODEL_CONFIG['num_neighbors'], 
        num_clusters=MODEL_CONFIG['num_clusters'],
        random_seed=MODEL_CONFIG['random_seed']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params} total")
    logger.info(f"Random seed: {MODEL_CONFIG['random_seed']}")
    return model


def train_model(model, X):
    """
    Train the ETE cluster model.
    
    Args:
        model: The ETEClusterModel to train
        X: Input tensor data
        
    Returns:
        torch.Tensor: Final cluster labels
    """
    logger.info("Starting training...")
    optimizer = optim.Adam(
        params=model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate']
    )
    for epoch in range(TRAINING_CONFIG['max_epochs']):
        model.train()
        optimizer.zero_grad()
    
        labels, spectral_loss, ortho_loss, cluster_loss = model(X)
        total_loss = spectral_loss + cluster_loss + ortho_loss
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == TRAINING_CONFIG['max_epochs'] - 1:
            logger.info(f"Epoch {epoch:3d}/{TRAINING_CONFIG['max_epochs']} - "
                       f"Loss: {total_loss.item():.4f} "
                       f"(Spectral: {spectral_loss.item():.4f}, "
                       f"Ortho: {ortho_loss.item():.4f}, "
                       f"Cluster: {cluster_loss.item():.4f})")
    
    logger.info("Training completed!")
    return labels


def save_community_assignments(df, labels):
    """
    Save community assignments to CSV file.
    
    Args:
        df: Original dataframe
        labels: Cluster labels from model
    """
    logger.info("Saving community assignments...")
    
    # Drop specified columns
    df_output = df.copy()
    for col in DATA_CONFIG['drop_columns']:
        if col in df_output.columns:
            df_output = df_output.drop(col, axis=1)
            logger.info(f"Dropped column: {col}")

    unique_vins = df_output['vin'].unique()
    for vin in unique_vins:
        community_label = int(labels[unique_vins == vin][0])
        df_output.loc[df_output['vin'] == vin, 'community'] = community_label

    output_filename = OUTPUT_CONFIG['community_csv_pattern'].format(
        method=DATA_CONFIG['method'],
        seed=MODEL_CONFIG['random_seed'],
        max_clusters=MODEL_CONFIG['num_clusters']
    )
    
    df_output.to_csv(output_filename, index=False)
    logger.info(f"Saved community assignments to: {output_filename}")


def create_community_heatmaps(X, labels):
    """
    Create heatmap visualizations for each community.
    
    Args:
        X: Input tensor data
        labels: Cluster labels
    """
    logger.info("Creating community heatmaps...")
    
    weeks_display = VIZ_CONFIG['weeks_to_display']
    total_weeks = X.shape[1]
    
    for label in np.unique(labels):
        logger.info(f"Creating heatmap for community {label}")
        community_data = X[labels == label].squeeze().numpy()
        if len(community_data.shape) == 1:
            reshaped_data = community_data.reshape(1, -1)[:, :weeks_display]
        else:
            reshaped_data = community_data[:, :weeks_display]
        
        fig, ax = plt.subplots(1, 1, figsize=VIZ_CONFIG['figure_size_heatmap'])
        im = ax.imshow(reshaped_data)
        
        ax.set_xlabel('Week', fontsize=VIZ_CONFIG['font_size_label'])
        ax.set_ylabel('Vehicle Number', fontsize=VIZ_CONFIG['font_size_label'])
    
        unique_vals, counts = np.unique(reshaped_data.flatten(), return_counts=True)
        colors = [im.cmap(im.norm(value)) for value in unique_vals]
        patches = [mpatches.Patch(color=colors[i], label=f'Cluster {int(unique_vals[i])}') 
                  for i in range(len(unique_vals))]
        
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(f'Community {label}', fontsize=VIZ_CONFIG['font_size_title'])

        filename = os.path.join('figures', 'heatmaps', OUTPUT_CONFIG['heatmap_pattern'].format(
            label=label, 
            format=VIZ_CONFIG['output_format']
        ))
        plt.savefig(filename, bbox_inches='tight', dpi=VIZ_CONFIG['dpi'])
        plt.close()
        
        logger.info(f"Saved heatmap: {filename}")


def create_community_histograms(X, labels):
    """
    Create histogram visualizations for each community showing cluster distributions.
    
    Args:
        X: Input tensor data
        labels: Cluster labels
    """
    logger.info("Creating community histograms...")
    
    weeks_display = VIZ_CONFIG['weeks_to_display']
    
    for label in np.unique(labels):
        logger.info(f"Creating histogram for community {label}")
        community_data = X[labels == label].squeeze().numpy()
        
        if len(community_data.shape) == 1:
            reshaped_data = community_data.reshape(1, -1)[:, :weeks_display]
        else:
            reshaped_data = community_data[:, :weeks_display]
        
        unique_vals, counts = np.unique(reshaped_data.flatten(), return_counts=True)
        mask = unique_vals > -1  # Exclude missing data
        unique_vals_filtered = unique_vals[mask]
        counts_filtered = counts[mask]

        fig, ax = plt.subplots(1, 1, figsize=VIZ_CONFIG['figure_size_histogram'])
        
        ax.bar(unique_vals_filtered, height=counts_filtered, 
               color=VIZ_CONFIG['bar_color'])
        ax.set_xlabel('Level One Cluster Label', fontsize=VIZ_CONFIG['font_size_label'])
        ax.set_ylabel(f'Frequency in Community {label}', fontsize=VIZ_CONFIG['font_size_label'])
        ax.tick_params(axis='both', which='major', labelsize=VIZ_CONFIG['font_size_tick'])

        filename = os.path.join('figures', 'histograms', OUTPUT_CONFIG['histogram_pattern'].format(
            label=label,
            format=VIZ_CONFIG['output_format']
        ))
        plt.savefig(filename, bbox_inches='tight', dpi=VIZ_CONFIG['dpi'])
        plt.close()
        
        logger.info(f"Saved histogram: {filename}")


def create_community_count_histogram(labels):
    """
    Create histogram showing the distribution of vehicles across communities.
    
    Args:
        labels: Cluster labels
    """
    logger.info("Creating community count histogram...")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    community_labels = [str(i) for i in range(len(unique_labels))]
    
    fig, ax = plt.subplots(1, 1, figsize=VIZ_CONFIG['figure_size_community_count'])
    
    ax.bar(range(len(unique_labels)), height=counts, color=VIZ_CONFIG['bar_color'])
    ax.set_xlabel('Community', fontsize=VIZ_CONFIG['font_size_label'])
    ax.set_ylabel('Number of VINs', fontsize=VIZ_CONFIG['font_size_label'])
    ax.set_xticks(range(len(unique_labels)), labels=community_labels, 
                  fontsize=VIZ_CONFIG['font_size_tick'])
    ax.tick_params(axis='y', which='major', labelsize=VIZ_CONFIG['font_size_tick'])
    
    filename = os.path.join('figures', 'community_counts', OUTPUT_CONFIG['community_count_file'].format(
        format=VIZ_CONFIG['output_format']
    ))
    plt.savefig(filename, bbox_inches='tight', dpi=VIZ_CONFIG['dpi'])
    plt.close()
    
    logger.info(f"Saved community count histogram: {filename}")
    logger.info(f"Community distribution: {dict(zip(unique_labels, counts))}")

def main():
    """
    Main execution function.
    """
    logger.info("=== ETE Cluster Model Clustering ===")
    if not os.path.exists("results"):
        os.makedirs("results")
        logger.info("Created results directory")
    try:
        create_output_directories()
        X, df = load_and_prepare_data()
        model = initialize_model()
    
        raw_labels = train_model(model, X)
        labels = raw_labels.squeeze().detach().numpy().argmax(axis=1)
        logger.info(f"Final cluster assignments shape: {labels.shape}")
        
        save_community_assignments(df, labels)
        create_community_heatmaps(X, labels)
        create_community_histograms(X, labels)
        create_community_count_histogram(labels)
        
        logger.info("=== Clustering completed successfully! ===")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()