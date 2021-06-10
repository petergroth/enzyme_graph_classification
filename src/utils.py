from pathlib import Path
from torch_geometric.datasets import TUDataset
import torch
import numpy as np

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
    
    # Shuffle
    torch.manual_seed(seed)

    # Split data
    split_idx = np.cumsum([int(len(dataset)*prop) for prop in props])
    data_train = dataset[:split_idx[0]]
    data_val = dataset[split_idx[0]:split_idx[1]]
    data_test = dataset[split_idx[1]:]

    return data_train, data_val, data_test
    
if __name__ == '__main__':
    load_data()