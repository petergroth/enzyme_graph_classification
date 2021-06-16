import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torchmetrics import Accuracy, MetricCollection, Precision, Recall


class GNN(nn.Module):
    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        hidden_size: int = 32,
        global_pooling: str = "global_mean_pool",
    ):
        super(GNN, self).__init__()
        self.n_node_features = n_node_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size

        self.gcn1 = GCNConv(
            in_channels=self.n_node_features,
            out_channels=self.hidden_size,
            normalize=True,
        )
        self.gcn2 = GCNConv(
            in_channels=self.hidden_size, out_channels=self.hidden_size, normalize=True
        )
        self.gcn3 = GCNConv(
            in_channels=self.hidden_size, out_channels=self.hidden_size, normalize=True
        )
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=self.n_classes)

        self.activation = nn.LeakyReLU()

        if global_pooling not in [
            "global_mean_pool",
            "global_add_pool",
            "global_max_pool",
        ]:
            raise ValueError(
                "Invalid global pooling. Must be one of ['global_mean_pool', 'global_add_pool', "
                "'global_max_pool']."
            )
        self.global_pooling = eval(global_pooling)

    # TODO: Consider different hidden_size for gcn vs fc1
    def forward(self, x: torch.tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        # Perform message passing
        x = self.activation(self.gcn1(x, edge_index))  # [nodes_in_batch, hidden_size]
        x = self.activation(self.gcn2(x, edge_index))  # [nodes_in_batch, hidden_size]
        x = self.gcn3(x, edge_index)  # [nodes_in_batch, hidden_size]
        # Apply global pooling layer
        embed = self.global_pooling(x, batch)  # [batch_size, hidden_size]
        # Feed through linear layer for prediction
        embed = self.fc1(embed)  # [batch_size, n_classes]

        return embed
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ConvNet')
        parser.add_argument('--hidden_size', type=int, default=32)
        parser.add_argument(
            '--global_pooling',
            choices=["global_mean_pool", "global_add_pool", "global_max_pool"],
            default="global_mean_pool")
    
        return parent_parser
    
    @staticmethod
    def from_argparse_args(namespace):
        ns_dict = vars(namespace)
        args = {
            'hidden_size': ns_dict.get('hidden_size', 32),
            'global_pooling': ns_dict.get(
                'global_pooling', 'global_mean_pooling')
            }
        
        return args


class GraphClassifier(pl.LightningModule):
    def __init__(self, model, lr: float = 3e-4):
        super().__init__()
        self.model = model
        self.num_classes = model.n_classes
        self.lr = lr

        # Setup metrics
        metrics = MetricCollection(
            [
                Accuracy(num_classes=self.num_classes),
                Precision(num_classes=self.num_classes),
                Recall(num_classes=self.num_classes),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        logits = self.model(batch.x, batch.edge_index, batch.batch)
        labels = batch.y
        loss = F.cross_entropy(logits, labels)
        # Logging
        log_output = self.train_metrics(F.softmax(logits, dim=-1), labels)
        self.log_dict(log_output, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch.x, batch.edge_index, batch.batch)
        labels = batch.y
        loss = F.cross_entropy(logits, labels)
        # Logging
        log_output = self.valid_metrics(F.softmax(logits, dim=-1), labels)
        self.log_dict(log_output, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return

    def test_step(self, batch, batch_idx):
        logits = self.model(batch.x, batch.edge_index, batch.batch)
        labels = batch.y
        loss = F.cross_entropy(logits, labels)
        # Logging
        log_output = self.test_metrics(F.softmax(logits, dim=-1), labels)
        self.log_dict(log_output, on_step=True, on_epoch=True)
        self.log("test_loss", loss)
        return self.test_metrics

    def forward(self, batch, batch_idx):
        # Makes the Classifier callable
        return self.model(batch.x, batch.edge_index, batch.batch)

    def predict_step(self, batch, batch_idx=None):
        logits = self.model(batch.x, batch.edge_index, batch.batch)
        ps = F.softmax(logits, dim=-1)
        ps = ps.max(1)[1]
        ps = ps.numpy()
        return ps
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ConvNet')
        parser.add_argument('--lr', type=float, default=3e-4)

        return parent_parser
    
    @staticmethod
    def from_argparse_args(namespace):
        ns_dict = vars(namespace)
        args = {
            'lr': ns_dict.get('lr', 3e-4),}
        
        return args