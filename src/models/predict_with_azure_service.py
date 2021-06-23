import argparse
import json

import requests
import torch
from numpy import genfromtxt
from torch_geometric.transforms import NormalizeFeatures
from torch.nn.functional import softmax

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_table_file", type=argparse.FileType("r"))
    parser.add_argument("node_attributes_file", type=argparse.FileType("r"))
    parser.add_argument("azure_endpoint", type=str)
    parser.add_argument(
        "--omit_row_normalization", dest="normalize", action="store_false"
    )

    args = parser.parse_args()

    return args


def load_graph_data(edge_table_file, node_attributes_file, normalize=True):
    with edge_table_file:
        edge_table = genfromtxt(edge_table_file, delimiter=",")
        # Index nodes from 0 and tranpose (size [2, num_edges])
        edge_table = (edge_table - edge_table.min()).T

    with node_attributes_file:
        node_attributes = genfromtxt(node_attributes_file, delimiter=",")

    if normalize:
        x = torch.from_numpy(node_attributes).to(torch.float)
        data = type("Data", (), {"x": x})
        node_attributes = NormalizeFeatures()(data).x.numpy()

    batch = [0 for n in range(len(node_attributes))]
    edge_table = edge_table.tolist()
    node_attributes = node_attributes.tolist()

    return node_attributes, edge_table, batch


def main():
    args = parser()
    node_attributes, edge_table, batch = load_graph_data(
        args.edge_table_file, args.node_attributes_file
    )

    # Convert the array to a serializable list in a JSON document
    input_json = json.dumps(
        {"x": node_attributes, "edge_index": edge_table, "batch": batch}
    )

    # Set the content type
    headers = {"Content-Type": "application/json"}

    predictions = requests.post(args.azure_endpoint, input_json, headers=headers)
    predicted_classes = json.loads(predictions.json())

    print(softmax(torch.Tensor(predicted_classes), dim=-1).tolist())


if __name__ == "__main__":
    main()
