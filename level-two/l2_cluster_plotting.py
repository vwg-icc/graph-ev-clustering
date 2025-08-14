#!/usr/bin/env python3
"""
Clustering analysis and visualization script (distribution + seasonal cluster transitions).

Usage:
    python l2_cluster_plotting.py results/<data.csv> --plots all --output-dir figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict

# Color palettes
COLOR_MAP = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', 
             '#CC6677', '#AA4499', '#882255', '#352334', '#FF4499', '#AA9944']
LIGHTER_PALETTE = ['#8899DD', '#55AA77', '#77CCBB', '#AADDEE', '#EEDD99', 
                   '#DD9999', '#CC77BB', '#BB5577', '#775577', '#FF77BB', '#CCBB77']

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
    df_cleaned = df[df['cluster'] != -1].copy()
    df_cleaned.dropna(subset=['cluster'], inplace=True)
    return df_cleaned

def plot_distribution(df_cleaned, save_path=None):
    plt.figure(figsize=(10, 4))
    n_clusters = df_cleaned['cluster'].nunique()
    sns.countplot(data=df_cleaned, x='community', hue='cluster', palette=COLOR_MAP[:n_clusters])
    plt.xlabel('Community', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.xticks(fontsize=13, rotation=0)
    plt.legend(title='Cluster')
    plt.title('Community-wise Cluster Distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"

def plot_seasonality(df_cleaned, save_path=None, top_n=5):
    """Plot top N cluster transitions per community per season as stacked bar charts."""
    # Create third-order cluster transitions
    df_sorted = df_cleaned.sort_values(['vin', 'week'])
    df_sorted['next_cluster'] = df_sorted.groupby('vin')['cluster'].shift(-1)
    df_transitions = df_sorted.dropna(subset=['next_cluster']).copy()
    df_transitions['transition'] = df_transitions['cluster'].astype(str) + ' -> ' + df_transitions['next_cluster'].astype(str)

    # Remove trivial transitions
    df_transitions = df_transitions[~df_transitions['transition'].str.match(r'^(\d+) -> \1 -> \1 -> \1$')]

    # Ensure date is datetime and assign seasons
    df_transitions['date'] = pd.to_datetime(df_transitions['date'])
    df_transitions['season'] = df_transitions['date'].apply(get_season)

    # Prepare colors for transitions
    unique_transitions = df_transitions['transition'].unique()
    color_map = {unique_transitions[i]: LIGHTER_PALETTE[i % len(LIGHTER_PALETTE)] for i in range(len(unique_transitions))}

    # Plot top-N transitions per community for each season
    fig, ax = plt.subplots(2, 2, figsize=(14.3, 10))
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    axes_positions = [(0,0), (0,1), (1,0), (1,1)]
    handles, labels = [], []

    for i, season in enumerate(seasons):
        axes = ax[axes_positions[i]]
        df_season = df_transitions[df_transitions['season'] == season]

        # Count transitions per community
        transition_counts = df_season.groupby(['community','transition']).size().reset_index(name='count')
        # top_transitions = transition_counts.groupby('community', group_keys=False).apply(lambda x: x.nlargest(top_n, 'count')).reset_index(drop=True)
        top_transitions = (
                        transition_counts
                        .sort_values(['community', 'count'], ascending=[True, False])
                        .groupby('community', group_keys=False)
                        .head(top_n)
                    )

        # Pivot and normalize
        transition_dist = top_transitions.pivot(index='community', columns='transition', values='count').fillna(0)
        transition_dist = transition_dist.div(transition_dist.sum(axis=1), axis=0)

        colors = [color_map[t] for t in transition_dist.columns]
        transition_dist.plot(ax=axes, kind='bar', stacked=True, color=colors, legend=None)
        axes.set_xlabel("Community", fontsize=14)
        axes.set_ylabel("Proportion of Transitions", fontsize=14)
        axes.set_title(f"{chr(97+i)}) {season}", loc='left', fontsize=15)
        axes.set_xticklabels(axes.get_xticklabels(), rotation=0, fontsize=13)
        # axes.set_yticklabels(np.round(axes.get_yticks(),2), fontsize=13)
        axes.yaxis.set_major_locator(MaxNLocator(integer=False))  # optional: control number of ticks
        axes.tick_params(axis='y', labelsize=13)

        h, l = axes.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Global legend
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(title="Cluster Transition", handles=by_label.values(), labels=by_label.keys(),
               loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate clustering analysis plots')
    parser.add_argument('input_file', help='Path to the CSV file with clustering results')
    parser.add_argument('--plots', nargs='+', choices=['distribution','seasonality','all'],
                        default=['all'], help='Types of plots to generate')
    parser.add_argument('--output-dir', default='figures/', help='Directory to save plots')
    parser.add_argument('--no-save', action='store_true', help='Display plots without saving')
    parser.add_argument('--format', default='pdf', choices=['pdf','png','jpg','svg'], help='Output format')
    
    args = parser.parse_args()
    df_cleaned = load_and_clean_data(args.input_file)
    print(f"Loaded {len(df_cleaned)} records with {df_cleaned['cluster'].nunique()} clusters")
    
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    plot_types = args.plots if 'all' not in args.plots else ['distribution','seasonality']
    
    for plot_type in plot_types:
        print(f"Generating {plot_type} plot...")
        save_path = None
        if not args.no_save:
            plot_dir = os.path.join(args.output_dir, plot_type)
            os.makedirs(plot_dir, exist_ok=True)
            save_path = os.path.join(plot_dir, f"{plot_type}.{args.format}")
        
        if plot_type == 'distribution':
            plot_distribution(df_cleaned, save_path)
        elif plot_type == 'seasonality':
            plot_seasonality(df_cleaned, save_path)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
