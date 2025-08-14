import torch.nn as nn


class Encoder(nn.Module):
    """
    LSTM-based Encoder for time-series data.

    Args:
        seq_len (int): Length of the input sequence.
        n_features (int): Number of features in each timestep.
        embedding_dim (int, optional): Size of the latent embedding. Default is 64.

    Architecture:
        - LSTM1: n_features → 2 * embedding_dim
        - LSTM2: hidden_dim → embedding_dim
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_features).

        Returns:
            Tensor: Encoded latent representation of shape (batch_size, embedding_dim).
        """
        x = x.reshape(-1, self.seq_len, self.n_features)
        x, _ = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape(-1, self.embedding_dim)


class Decoder(nn.Module):
    """
    LSTM-based Decoder for time-series data reconstruction.

    Args:
        seq_len (int): Length of the output sequence.
        input_dim (int, optional): Size of the latent embedding. Default is 64.
        n_features (int, optional): Number of features in each timestep. Default is 1.

    Architecture:
        - LSTM1: input_dim → input_dim
        - LSTM2: input_dim → hidden_dim
        - Linear: hidden_dim → n_features
    """

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        """
        Forward pass for the decoder.

        Args:
            x (Tensor): Latent representation of shape (batch_size, input_dim).

        Returns:
            Tensor: Reconstructed sequence of shape (batch_size, seq_len, n_features).
        """
        x = x.repeat(self.seq_len, 1)
        x = x.reshape(-1, self.seq_len, self.input_dim)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x.reshape(-1, self.seq_len, self.hidden_dim)
        return self.output_layer(x)
