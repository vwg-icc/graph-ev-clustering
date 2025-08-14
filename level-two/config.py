# Configuration file for ETE Cluster Model Training


DATA_CONFIG = {
    'csv_file': '../level-one/files/preprocessed/VINS_50/10min/kmedoids/K_6_distance_manhattan_random_0/output.csv',
    'method': 'kmedoids',  # 'kmeans' or 'kmedoids'
    'index_column': 'vin',
    'columns': 'week',
    'values': 'cluster',
    'fill_na_value': -2,
    'drop_columns': ['covid']  # columns to drop before saving
}

MODEL_CONFIG = {
    'dim': 1,
    'encoder_size': 128,
    'num_neighbors': 15,
    'num_clusters': 5,
    'random_seed': 11
}

TRAINING_CONFIG = {
    'max_epochs': 75,
    'learning_rate': 0.01
}

VIZ_CONFIG = {
    'figure_size_heatmap': (10, 10),
    'figure_size_histogram': (12, 8),
    'figure_size_community_count': (12, 6),
    'dpi': 150,
    'colormap': 'viridis',
    'bar_color': '#88CCEE',
    'font_size_title': 18,
    'font_size_label': 15,
    'font_size_tick': 13,
    'weeks_to_display': 52,  # first N weeks to display in heatmaps
    'output_format': 'pdf'  # 'pdf' or 'png'
}

OUTPUT_CONFIG = {
    'community_csv_pattern': 'results/community_{method}_k_6_random_seed_{seed}_max_{max_clusters}.csv',
    'heatmap_pattern': 'Community_{label}_heatmap.{format}',
    'histogram_pattern': 'Community_{label}_histogram.{format}',
    'community_count_file': 'community_count_histogram.{format}'
}