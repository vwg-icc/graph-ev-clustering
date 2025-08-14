import torch
import torch_geometric
from torch_geometric.nn import knn_graph, GATConv, ClusterGCNConv
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.nn import DenseGraphConv, DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch

class ETEClusterModel(torch.nn.Module):
    """
    Encoder-Temporal-Embedding (ETE) Clustering Model.

    Combines LSTM-based encoding with graph-based clustering using DMoNPooling.

    Args:
        dim (int): Input feature dimension.
        encoder_size (int): Hidden size of the LSTM encoder.
        num_neighbors (int): Number of neighbors for kNN graph construction.
        num_clusters (int): Number of clusters for DMoN pooling.
        random_seed (int): Random seed for reproducibility.
    """

    def __init__(self, dim, encoder_size, num_neighbors, num_clusters, random_seed):
        super(ETEClusterModel, self).__init__()

        torch_geometric.seed_everything(random_seed)

        self.lstm = torch.nn.LSTM(dim, encoder_size, batch_first=True, num_layers=1)
        self.pool = DMoNPooling(channels=encoder_size, k=num_clusters)
        self.GCN = ClusterGCNConv(encoder_size, encoder_size)
        self.num_neighbors = num_neighbors

    def encode(self, x, batch=None):
        """
        Encode input sequences into latent embeddings using LSTM.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).
            batch (Tensor, optional): Batch vector for graph operations.

        Returns:
            Tensor: Hidden state from LSTM of shape (1, batch_size, encoder_size).
        """
        output, (h_n, c_n) = self.lstm(x)
        return h_n

    def build_graph(self, x, batch=None):
        """
        Construct a k-NN graph from node embeddings.

        Args:
            x (Tensor): Node embeddings of shape (num_nodes, embedding_dim).
            batch (Tensor, optional): Batch vector for multiple graphs.

        Returns:
            Data: PyG Data object containing node features and edge_index.
        """
        edge_index = knn_graph(x.squeeze(), k=self.num_neighbors, batch=batch, loop=False)
        return Data(x, edge_index)

    def build_temporal_graph(self, x):
        """
        Placeholder for temporal graph construction (currently not implemented).

        Args:
            x (Tensor): Node embeddings.

        Returns:
            None
        """
        return None

    def cluster(self, x):
        """
        Apply DMoN pooling to compute clusters.

        Args:
            x (Data): PyG Data object with node features and edge_index.

        Returns:
            tuple: Cluster assignments, spectral loss, orthogonality loss, cluster loss.
        """
        s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x.x, to_dense_adj(x.edge_index))
        return s, spectral_loss, ortho_loss, cluster_loss

    def forward(self, inputs):
        """
        Forward pass through the full model.

        Steps:
            1. Encode input sequences.
            2. Build k-NN graph from embeddings.
            3. Apply GCN layer with ReLU activation.
            4. Apply DMoN pooling to produce clusters.

        Args:
            inputs (Tensor): Input sequences of shape (batch_size, seq_len, dim).

        Returns:
            tuple: Cluster labels, spectral loss, orthogonality loss, cluster loss.
        """
        enc = self.encode(inputs)
        G = self.build_graph(enc)
        G.x = self.GCN(G.x, G.edge_index)
        G.x = torch.nn.ReLU()(G.x)
        labels, spectral_loss, ortho_loss, cluster_loss = self.cluster(G)
        return labels, spectral_loss, ortho_loss, cluster_loss
