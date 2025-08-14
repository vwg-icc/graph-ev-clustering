import torch
import torch_geometric
from torch_geometric.nn import knn_graph, GATConv, ClusterGCNConv
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.nn import DenseGraphConv, DMoNPooling, GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


class ETEClusterModel(torch.nn.Module):
    def __init__(
        self,
        dim,
        encoder_size,
        num_neighbors,
        num_clusters,
        random_seed

    ):
        super(ETEClusterModel, self).__init__()

        torch_geometric.seed_everything(random_seed)

        self.lstm = torch.nn.LSTM(dim, encoder_size, batch_first=True, num_layers=1)

        # self.trans = torch.nn.TransformerEncoderLayer(dim, nhead=1, dim_feedforward=encoder_size, batch_first=True)

        self.pool = DMoNPooling(channels=encoder_size, k=num_clusters)

        self.GCN = ClusterGCNConv(encoder_size, encoder_size)

        self.num_neighbors = num_neighbors

        


    def encode(self, x, batch=None):

        output, (h_n, c_n) = self.lstm(x)

        # h_n = self.trans(x)

        return h_n
        

    def build_graph(self, x, batch=None):

        edge_index = knn_graph(x.squeeze(), k=self.num_neighbors, batch=batch, loop=False)

        return Data(x, edge_index)

    def build_temporal_graph(self, x):

        # edge_index = knn_graph(x, k=2, batch=batch, loop=False)

        return None

    def cluster(self, x):

         s, out, out_adj, spectral_loss, ortho_loss, cluster_loss = self.pool(x.x, to_dense_adj(x.edge_index))

         return s, spectral_loss, ortho_loss, cluster_loss

    def forward(self, inputs):

        enc = self.encode(inputs)

        # enc = torch.nn.ReLU()(enc)

        G = self.build_graph(enc)

        G.x = self.GCN(G.x, G.edge_index)

        G.x = torch.nn.ReLU()(G.x)
        
        labels, spectral_loss, ortho_loss, cluster_loss = self.cluster(G)

        return labels, spectral_loss, ortho_loss, cluster_loss