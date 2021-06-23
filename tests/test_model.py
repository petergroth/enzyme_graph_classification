import pytest
import torch
from torch_geometric.data import Batch

from src.models.model import GNN, GraphClassifier


def test_initialization_pooling():
    with pytest.raises(ValueError):
        # Try with invalid pooling
        GNN(10, 10, 32, global_pooling="global_mean_pol")


def test_forward_pass_batch_size():
    model = GNN(10, 10, 32)
    with pytest.raises(ValueError):
        # Try with invalid pooling
        x = torch.rand(100, 10)
        edge_index = torch.randint(100, (2, 200))
        batch = torch.randint(50, (101,))
        _ = model.forward(x, edge_index, batch)


def test_forward_pass_n_features():
    # Create model with 10 features
    model = GNN(10, 10, 32)
    with pytest.raises(ValueError):
        # Forward pass with 11 features
        num_node_feature = 11
        num_nodes = 100
        num_graph = 10
        x = torch.rand(num_nodes, num_node_feature)
        edge_index = torch.randint(num_nodes, (2, 200))
        batch = torch.randint(num_graph, (num_nodes,))
        _ = model.forward(x, edge_index, batch)


def test_forward_pass():
    num_class = 10
    num_node_feature = 10
    num_nodes = 100
    num_graph = 10
    model = GNN(n_node_features=num_node_feature, n_classes=num_class)
    x = torch.rand(num_nodes, num_node_feature)
    edge_index = torch.randint(num_nodes, (2, 200))
    batch = torch.randint(num_graph, (num_nodes,))
    output = model.forward(x, edge_index, batch)
    assert list(output.shape) == [num_graph, num_class]


def test_training_step():
    num_class = 10
    num_node_feature = 10
    num_graph = 10
    num_nodes = 100
    model = GNN(n_node_features=num_node_feature, n_classes=num_class)
    classifier = GraphClassifier(model=model)
    batch = Batch(
        batch=torch.randint(num_graph, (num_nodes,)),
        x=torch.rand(num_nodes, num_node_feature),
        edge_index=torch.randint(num_nodes, (2, 200)),
        y=torch.randint(num_class, (num_graph,)),
        ptr=torch.arange(num_graph + 1),
    )
    # Create batch
    loss = classifier.training_step(batch, 0)

    assert type(loss.item()) == float
