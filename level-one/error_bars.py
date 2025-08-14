"""
Run error bar clustering experiments using parameters from a config JSON and CLI arguments.

Usage:
    python run_error_bar_experiments.py --config config.json \
        --k_clusters 5 6 7 8 9 \
        --random_seed_vals 0 1 2 3 4 5 6 7 8 9 \
        --n_jobs 4 \
        --save_csv True \
        --output_dir ./results/
"""

import argparse
import logging
import random
import os
import time
import numpy as np
import pandas as pd
from CreateCluster import CreateCluster
from utils import create_data_array
from parse_config import Config
from joblib import Parallel, delayed
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def cluster_fn(data, params):
    cluster = CreateCluster(params)
    return cluster.cluster_data(data)

def generate_error_bars(params,
                        data_array,
                        cluster_fn,
                        k_clusters=None,
                        random_seed_vals=None,
                        n_jobs=4,
                        save_csv=True,
                        output_dir="./"):
    if k_clusters is None:
        k_clusters = getattr(params, 'k_clusters', [5, 6, 7, 9])
    if random_seed_vals is None:
        random_seed_vals = getattr(params, 'random_seed_vals', [0, 1, 2, 3, 4, 5])

    logger.info(f"Generating error bars for k_clusters={k_clusters}, seeds={random_seed_vals}")

    param_combinations = []
    for seed in random_seed_vals:
        for k in k_clusters:
            p = deepcopy(params)
            p.n_clusters = k
            p.random_state = seed
            param_combinations.append(p)

    def run_single(params):
        silhouette, inertia, dbi, k_val, seed_val = None, None, None, None, None
        try:
            labels, cluster_res = cluster_fn(data_array, params)
            silhouette = cluster_res.get("Silhouette score")
            inertia = cluster_res.get("Sum of distances:")
            dbi = cluster_res.get("Davies-Bouldin index:")
            k_val = params.n_clusters
            seed_val = params.random_state
        except Exception as e:
            logger.error(f"Clustering failed for K={params.n_clusters}, seed={params.random_state}: {e}")
        return silhouette, inertia, dbi, k_val, seed_val

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single)(p) for p in param_combinations
    )

    silhouette = []
    inertia = []
    dbi = []
    k_vals = []
    seeds = []

    for sil, iner, d, k_val, seed_val in results:
        silhouette.append(sil)
        inertia.append(iner)
        dbi.append(d)
        k_vals.append(k_val)
        seeds.append(seed_val)

    df = pd.DataFrame({
        "K": k_vals,
        "seed": seeds,
        "silhouette": silhouette,
        "dbi": dbi,
        "inertia": inertia
    })

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        csv_name = f"error_data_{params.cluster_algo}_{params.distance_metric}_{k_clusters}_{random_seed_vals}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        df.to_csv(csv_path, index=False)
        logger.info(f"Error bars results saved to {csv_path}")

    return df

def main(args):
    logger.info("Starting error bar experiment")

    params = Config.from_file(args.config)
    random.seed(params.random_state)
    np.random.seed(params.random_state)

    logger.info(f"Using features from: {params.features_dir}")
    train_raw_ns, *_ = create_data_array(params.features_dir)
    logger.info(f"Loaded data | Shape: {train_raw_ns.shape}")

    error_bars_cfg = {
        "k_clusters": args.k_clusters,
        "random_seed_vals": args.random_seed_vals,
        "n_jobs": args.n_jobs,
        "save_csv": args.save_csv,
        "output_dir": args.output_dir,
    }
    logger.info(f"Using error bars config: {error_bars_cfg}")

    generate_error_bars(
        params=params,
        data_array=train_raw_ns,
        cluster_fn=lambda data, p=params: cluster_fn(data, p),
        k_clusters=error_bars_cfg["k_clusters"],
        random_seed_vals=error_bars_cfg["random_seed_vals"],
        n_jobs=error_bars_cfg["n_jobs"],
        save_csv=error_bars_cfg["save_csv"],
        output_dir=error_bars_cfg["output_dir"]
    )

    logger.info("Finished error bar experiment")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run error bar clustering experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to clustering config JSON")
    parser.add_argument("--k_clusters", nargs='+', type=int, default=[5, 6, 7, 8, 9], help="List of K values to try for clustering")
    parser.add_argument("--random_seed_vals", nargs='+', type=int, default=list(range(10)), help="List of random seeds to try")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument("--save_csv", type=bool, default=True, help="Save output CSV")
    parser.add_argument("--output_dir", type=str, default="./results/", help="Output directory")
    args = parser.parse_args()
    main(args)
