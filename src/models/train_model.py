#%%

from src.models.model import GNN
from src.utils import load_data

from torch_geometric.data import DataLoader

train_data, _, _ = load_data()

train_loader = DataLoader(train_data, shuffle=False, batch_size=64)
data = next(iter(train_loader))

x, edge_index, batch = data.x, data.edge_index, data.batch

model = GNN(train_data.num_node_features, 10, 32, 'global_mean_pool')
tmp = model(x, edge_index, batch)

