import requests
import json
import argparse
from numpy import genfromtxt


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('edge_table_file', type=argparse.FileType('r'))
    parser.add_argument('node_attributes_file', type=argparse.FileType('r'))
    parser.add_argument('azure_endpoint', type=str)
    
    args = parser.parse_args()

    return args


def load_graph_data(edge_table_file, node_attributes_file):
    with edge_table_file:
        edge_table = genfromtxt(edge_table_file, delimiter=',')
    
    with node_attributes_file:
        node_attributes = genfromtxt(node_attributes_file, delimiter=',')

    batch = [0 for n in range(len(node_attributes))]    
    edge_table = edge_table.tolist()
    node_attributes = node_attributes.tolist()

    return node_attributes, edge_table, batch


def main():
    args = parser()
    node_attributes, edge_table, batch = load_graph_data(
        args.edge_table_file, args.node_attributes_file)

    # Convert the array to a serializable list in a JSON document
    input_json= json.dumps(
        {'x': node_attributes,
         'edge_index': edge_table,
         'batch': batch})
    
    # Set the content type
    headers = {'Content-Type': 'application/json'}

    predictions = requests.post(
        args.azure_endpoint, input_json, headers=headers)
    predicted_classes = json.loads(predictions.json())
    
    print(predicted_classes)


if __name__ == '__main__':
    main()