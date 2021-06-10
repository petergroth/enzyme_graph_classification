from pathlib import Path
from re import split
from torch_geometric import data
from torch_geometric.datasets import TUDataset
import numpy as np
import torch
from torchvision import transforms

def load_data(props=[0.7, 0.15, 0.15], seed=42):
    # Data location
    project_dir = project_dir = Path(__file__).resolve().parents[1]
    data_dir = str(project_dir) + '/data/'

    # Load data
    dataset = TUDataset(
        root=data_dir,
        name='ENZYMES',
        use_node_attr=True,
        use_edge_attr=True)
    dataset = dataset.shuffle()
    num_node_labels = 3
    num_node_features = dataset.num_node_features - num_node_labels
    
    # Split data
    split_idx = np.cumsum([int(len(dataset)*prop) for prop in props])
    data_train = dataset[:split_idx[0]]
    data_val = dataset[split_idx[0]:split_idx[1]]
    data_test = dataset[split_idx[1]:]
    print(split_idx)
    # Normalize node attributes
    mu = (data_train.data.x[:,:num_node_features].mean(dim=0)).clone().detach()
    std = (data_train.data.x[:,:num_node_features].std(dim=0)).clone().detach()

    x = data_train.data.x[:,:num_node_features].clone().detach()
    y = data_val.data.x[:,:num_node_features].clone().detach()
    z = data_test.data.x[:,:num_node_features].clone().detach()

    data_train.data.x[:,:num_node_features] = ((x - mu) / std).clone().detach()
    data_val.data.x[:,:num_node_features] = ((y - mu) / std).clone().detach()
    data_test.data.x[:,:num_node_features] = ((z - mu) / std).clone().detach()
    
    print(data_train.data.x.mean(dim=0))
    print(data_train.data.x.std(dim=0))

    print(data_val.data.x.mean(dim=0))
    print(data_val.data.x.std(dim=0))

    print(data_val.data.x.mean(dim=0))
    print(data_val.data.x.std(dim=0))

    print(torch.eq(data_train.data.x.mean(dim=0), data_test.data.x.mean(dim=0)))
if __name__ == '__main__':
    load_data()