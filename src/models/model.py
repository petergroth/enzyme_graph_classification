import torch
from torch import nn
from torch_geometric.nn import (GCNConv, global_add_pool, global_max_pool,
                                global_mean_pool)


class GNN(nn.Module):
    def __init__(self, n_node_features: int, n_classes: int, hidden_size: int = 32, global_pooling: str = 'global_mean_pool'):
        super(GNN, self).__init__()
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        self.gcn1 = GCNConv(in_channels=self.n_node_features, out_channels=self.hidden_size, normalize=True)
        self.gcn2 = GCNConv(in_channels=self.hidden_size, out_channels=self.hidden_size, normalize=True)
        self.gcn3 = GCNConv(in_channels=self.hidden_size, out_channels=self.hidden_size, normalize=True)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.n_classes)

        self.activation = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)

        if global_pooling not in ['global_mean_pool', 'global_add_pool', 'global_max_pool']:
            raise ValueError("Invalid global pooling. Must be one of ['global_mean_pool', 'global_add_pool', "
                             "'global_max_pool'].")
        self.global_pooling = eval(global_pooling)

    # TODO: Add types for forward pass
    def forward(self, x: torch.tensor, edge_index, batch):
        # Perform message passing
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        x = self.gcn3(x, edge_index)
        # Apply global pooling layer
        embed = self.global_pooling(x, batch)
        # Feed through linear layer for prediction
        embed = self.fc1(embed)
        logits = self.log_softmax(embed)

        return logits