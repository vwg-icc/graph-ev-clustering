import random
import logging
import time
import os
import pandas as pd

from parse_config import Config
from CreateCluster import CreateCluster
from utils import (
    create_data_array,
    generate_output_df,
    create_non_drop_data_arr,
    get_train_vins
)
from plot_utils import plot_kde

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def cluster_fn(data, params):
    cluster = CreateCluster(params)
    return cluster.cluster_data(data)


if __name__ == "__main__":
    logger.info("Starting clustering pipeline")
    start = time.time()

    try:
        params = Config.from_file('config.json')
        random.seed(params.random_state)
        logger.info(f"Loaded config | Seed: {params.random_state} | Features dir: {params.features_dir}")

        # Load data
        train_raw_ns, valid_train_vins, drop_train_vins, valid_train_weeks, drop_train_weeks, scaler = create_data_array(params.features_dir)
        logger.info(f"Data loaded | Shape: {train_raw_ns.shape} | Valid VINs: {len(valid_train_vins)} | Dropped VINs: {len(drop_train_vins)}")

        # Run clustering
        cluster = CreateCluster(params)
        labels, cluster_res = cluster.cluster_data(train_raw_ns)
        logger.info(f"Clustering done | Metrics: {cluster_res}")

        # Save clustering results
        result_path = "clustering_results_all.csv"
        df = pd.DataFrame([cluster_res])
        if os.path.exists(result_path):
            df_res_clus = pd.read_csv(result_path)
            df_res_clus = pd.concat([df_res_clus, df], ignore_index=True)
        else:
            df_res_clus = df
        df_res_clus.to_csv(result_path, index=False)
        logger.info(f"Clustering results saved → {result_path}")

        # Training characterization
        train_vins = get_train_vins(params.features_dir)
        train_characterize = create_non_drop_data_arr(params.features_dir)
        logger.info(f"Training data prepared | # VINs: {len(train_vins)}")

        # Directory setup
        base_dir = os.path.join(params.features_dir, params.cluster_algo)
        base_dir = f"{base_dir}/K_{params.n_clusters}_distance_{params.distance_metric}_random_{params.random_state}"
        if params.cluster_algo == "agglomerative":
            base_dir += f"_{params.distance_threshold}"
        if params.dr:
            base_dir += "/DR/"
        os.makedirs(base_dir, exist_ok=True)

        # Save output
        output = generate_output_df(train_vins, params.features_dir, valid_train_vins,
                                    valid_train_weeks, drop_train_vins, drop_train_weeks, labels)
        output_filename = f"{base_dir}/output.csv"
        output.to_csv(output_filename, index=False)
        logger.info(f"Output saved → {output_filename} | Shape: {output.shape}")

        logger.info("Plotting and saving KDE plots")
        os.mkdir('Figures')
        plot_kde(params)
        logger.info(f"KDE plots saved")

        logger.info(f"Pipeline complete | Total time: {time.time() - start:.2f} sec")

    except Exception as e:
        logger.error("An error occurred during execution", exc_info=True)
        raise

    logger.info("Script finished")
