from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import TUDataset


def load_data(props=[0.7, 0.15, 0.15], seed=42):
    torch.manual_seed(seed)
    # Data location
    project_dir = project_dir = Path(__file__).resolve().parents[1]
    data_dir = str(project_dir) + '/data/'

    # Load data
    dataset = TUDataset(
        root=data_dir,
        name='ENZYMES',
        use_node_attr=True,
        use_edge_attr=True)
    
    # Shuffle
    dataset.shuffle()

    # Split data
    split_idx = np.cumsum([int(len(dataset)*prop) for prop in props])
    data_train = dataset[:split_idx[0]]
    data_val = dataset[split_idx[0]:split_idx[1]]
    data_test = dataset[split_idx[1]:]

    print(data_train.data.x)

    return data_train, data_val, data_test

if __name__ == '__main__':
    load_data()