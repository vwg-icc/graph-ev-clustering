import json
from dataclasses import dataclass


@dataclass
class Config:
    """
    A dataclass for storing configuration parameters.
    """
    features_dir: str
    figures_dir: str
    n_clusters: int
    cluster_algo: str
    max_iter: int
    random_state : int
    distance_metric: str
    precomputed : str
    experiment: int
    distance_threshold: int
    dr: bool

    @classmethod
    def from_file(cls, filename):
        """
        Constructs a Config object from a configuration file.

        Returns:
            Config: A Config object containing the configuration parameters.
        """
        # Read the configuration file.
        with open(filename, 'r') as f:
            config_dict = json.load(f)

        # Create a Config object from the configuration dictionary.
        return cls(**config_dict)
