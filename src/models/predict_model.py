import argparse
from src import project_dir
from numpy import genfromtxt
import torch
from src.models.model import GNN
from torch_geometric.transforms import NormalizeFeatures

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('edge_table_file', type=argparse.FileType('r'))
    parser.add_argument('node_attributes_file', type=argparse.FileType('r'))
    parser.add_argument(
        '--model_path', default=project_dir + '/models/model.pth', type=str)
    parser.add_argument(
        '--omit_row_normalization', dest='normalize', action='store_false')
    
    args = parser.parse_args()

    return args


def load_graph_data(edge_table_file, node_attributes_file, normalize=True):
    with edge_table_file:
        edge_table = genfromtxt(edge_table_file, delimiter=',')
        # Index nodes from 0 and tranpose (size [2, num_edges])
        edge_table = (edge_table - edge_table.min()).T
    
    with node_attributes_file:
        node_attributes = genfromtxt(node_attributes_file, delimiter=',')
    
    x = torch.from_numpy(node_attributes).to(torch.float)
    edge_index = torch.from_numpy(edge_table).to(torch.int64)
    batch = torch.zeros(len(node_attributes)).to(torch.int64)

    if normalize:
        data = type('Data', (), {'x': x})
        x = NormalizeFeatures()(data).x
       
    return x, edge_index, batch


def main():
    args = parser()
    x, edge_index, batch = load_graph_data(
        args.edge_table_file, args.node_attributes_file, args.normalize)

    model_dict = torch.load(args.model_path)
    model = GNN(**model_dict['model_kwargs'])
    model.load_state_dict(model_dict['state_dict'])
    model.eval()

    pred = model.forward(x, edge_index, batch).detach().squeeze()
    print(pred)


if __name__ == '__main__':
    main()
