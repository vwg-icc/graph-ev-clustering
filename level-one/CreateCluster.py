import numpy as np
from utils import get_distance_matrix
from tslearn.clustering import TimeSeriesKMeans
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import hdbscan
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from tslearn.barycenters import euclidean_barycenter
from constants import STORAGE_DIR

def cluster_fn(data, params):
    from CreateCluster import CreateCluster  # <-- moved here to avoid circular import
    cluster = CreateCluster(params)
    return cluster.cluster_data(data)

class CreateCluster:
    """
    A class for clustering data using various algorithms.

    Args:
        n_clusters (int): The number of clusters to form.
        init (str): The initialization method for the algorithm.
        max_iter (int): The maximum number of iterations for the algorithm.
        random_state (int): The random seed for the algorithm.
    """

    def __init__(self, params):
        self.n_clusters = params.n_clusters
        self.cluster_algo = params.cluster_algo
        self.max_iter = params.max_iter
        self.random_state = params.random_state
        self.distance_metric = params.distance_metric
        self.precomputed = params.precomputed
        self.dir = params.features_dir
        self.experiment = params.experiment
        self.distance_threshold = params.distance_threshold
        self.dr = params.dr

    def cluster_data(self, data):
        """
        Clusters the input data using a specified clustering method.

        Args:
            data (numpy.ndarray): The data to be clustered.
            self.cluster_algo (str): The name of the clustering method to use.

        Returns:
            numpy.ndarray: An array containing the labels assigned to each data point by the clustering algorithm.
        """
        if self.cluster_algo == 'kmeans':
            # Call the k-means clustering function.
            labels, cluster_obj = self.k_means(data)
            medoids = None
            sum_of_dis = round(cluster_obj.inertia_,3)

        elif self.cluster_algo == 'agglomerative':
            # Call the hierarchical clustering function.
            labels, cluster_obj, dend = self.agglomerative_clustering(data)
            medoids = 1
            sum_of_dis = round(np.max(np.max(dend['dcoord'], axis=1)) - np.min(np.max(dend['dcoord'], axis=1)),3)

        elif self.cluster_algo == 'kmedoids':
            # Call the k-medoids clustering function.
            labels, cluster_obj = self.k_medoids(data)
            medoids = cluster_obj.medoid_indices_
            sum_of_dis = round(cluster_obj.inertia_,3)

        else:
            # Handle unsupported clustering methods.
            raise ValueError(f'Unsupported clustering method {self.cluster_algo}')

        data_flat = data[:].reshape(data.shape[0], -1) # flat dataset
        intertia =  None if self.cluster_algo == "agglomerative" else round(cluster_obj.inertia_,3)
        cluster_results = {
            'num_clusters': self.n_clusters,
            'clustering_algorithm': self.cluster_algo,
            'distance measure': self.distance_metric,
            'Silhouette score':round(silhouette_score(data_flat,labels),3),
            'Sum of distances:' : intertia,
            'Davies-Bouldin index:':round(davies_bouldin_score(data_flat,labels),3),
            'medoids': medoids,
            'num_of_data_samples': len(data),
            'distance_threshold': self.distance_threshold,
            }

        return labels, cluster_results

    def k_medoids(self, data):
        """
        Clusters data using the k-medoids algorithm.

        Args:
            data (numpy.ndarray): A 2D array of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: A 1D array of shape (n_samples,) containing the cluster
                assignments for each data point.
        """
        data_flat = data[:].reshape(data.shape[0], -1)
        distance_matrix_np = get_distance_matrix(data_flat, self.distance_metric, self.dir)
        kmedoids = KMedoids(n_clusters=self.n_clusters, 
                metric=self.precomputed,
                method='pam',
                init='k-medoids++', 
                random_state=self.random_state).fit(distance_matrix_np) # choose distance
        labels = kmedoids.labels_ 
        return labels, kmedoids

    def k_means(self, data):
        """
        Clusters data using the k-means algorithm.

        Args:
            data (numpy.ndarray): A 2D array of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: A 1D array of shape (n_samples,) containing the cluster
                assignments for each data point.
        """
        # Define the k-means algorithm.
        data_flat = data[:].reshape(data.shape[0], -1) # flat dataset
        if self.precomputed:
            distance_matrix_np = get_distance_matrix(data_flat,self.distance_metric, self.dir)
        train_raw_s = data.copy()
        np.random.shuffle(train_raw_s) # shuffled dataset
        tskmeans = TimeSeriesKMeans(n_clusters=self.n_clusters, 
                                    metric=self.distance_metric, 
                                    random_state=self.random_state).fit(train_raw_s)



        labels = tskmeans.predict(data)

        return labels, tskmeans

    def agglomerative_clustering(self, data):
        """
        Clusters data using the agglomerative clustering algorithm.

        Args:
            data (numpy.ndarray): A 2D array of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: A 1D array of shape (n_samples,) containing the cluster
                assignments for each data point.
        """
        # Define the agglomerative clustering algorithm.

        model_name = 'agglomerative'
        print ("Running Agg clustering")
        data_flat = data[:].reshape(data.shape[0], -1) # flat dataset
        if self.n_clusters:
            agg = AgglomerativeClustering(n_clusters=self.n_clusters,
                                    affinity='euclidean',
                                    linkage='ward',
                                    ).fit(data_flat)
        else:

            agg = AgglomerativeClustering(n_clusters=self.n_clusters,
                                        affinity='euclidean',
                                        linkage='ward',
                                        distance_threshold=self.distance_threshold
                                        ).fit(data_flat)
        labels = agg.labels_
        optimal = len(set(labels))
        self.n_clusters = optimal
        linkage = shc.linkage(data_flat, method='ward', metric='euclidean')
        dend = shc.dendrogram(linkage, no_labels=True)

        return labels, agg, dend
