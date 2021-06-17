import argparse
from src import project_dir
from numpy import genfromtxt
import torch
from src.models.model import GraphClassifier, GNN

class Batch:
    def __init__(self, x, edge_index, batch):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('edge_table_file', type=argparse.FileType('r'))
    parser.add_argument('node_attributes_file', type=argparse.FileType('r'))
    parser.add_argument('model_path', type=str)
    
    args = parser.parse_args()

    return args


def load_graph_data(edge_table_file, node_attributes_file):
    with edge_table_file:
        edge_table = genfromtxt(edge_table_file, delimiter=',')
    
    with node_attributes_file:
        node_attributes = genfromtxt(node_attributes_file, delimiter=',')
    
    batch = Batch(
        x=torch.from_numpy(node_attributes),
        edge_index=torch.from_numpy(edge_table),
        batch=torch.zeros(len(node_attributes)))
    
    return batch


def main():
    args = parser()
    batch = load_graph_data(
        args.edge_table_file, args.node_attributes_file)
    import os
    print(os.path.isfile(args.model_path))

    model = GraphClassifier.load_from_checkpoint(args.model_path)

    pred = model.forward(batch)
    print(pred)


if __name__ == '__main__':
    main()
