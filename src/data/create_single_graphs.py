import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('edge_table_file', type=argparse.FileType('r'))
    parser.add_argument('graph_indicator_file', type=argparse.FileType('r'))
    parser.add_argument('node_attributes_file', type=argparse.FileType('r'))
    parser.add_argument('node_labels_file', type=argparse.FileType('r'))
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()

    return args


def write_graph(output_dir, graph_no, graph_attributes, graph_edges):

    with open(f'{output_dir}/graph_{graph_no}_node_attributes.txt', 'w') as f:
        f.write(graph_attributes)
    
    with open(f'{output_dir}/graph_{graph_no}_edges.txt', 'w') as f:
        f.write(graph_edges)
    

def get_edges(args, last_node):
    graph_edges = ''

    for line in args.edge_table_file:
        edge = line
        node_1, node_2 = line.split(', ')

        if int(node_1) <= last_node and int(node_2) <= last_node:
            graph_edges += edge
        else:
            break
    
    return graph_edges, edge

def one_hot(number):
    if number == '1':
        return '1, 0, 0'
    elif number == '2':
        return '0, 1, 0'
    elif number == '3':
        return '0, 0, 1'


def extract_graphs(args):
    graph_no = 1
    last_node_in_graph = 0
    graph_attributes = ''
    graph_edges = ''

    for line in args.graph_indicator_file:
        current_graph = int(line.strip())

        if current_graph == graph_no:
            last_node_in_graph += 1
            graph_attributes += args.node_attributes_file.readline().strip()
            graph_attributes += ', ' + one_hot(args.node_labels_file.readline().strip()) + '\n'
        
        else:
            # Finalize current graph
            edges_in_graph, first_edge_in_next_graph = get_edges(
                args, last_node_in_graph)
            graph_edges += edges_in_graph

            write_graph(args.output_dir, graph_no, graph_attributes, graph_edges)

            # Start next graph
            graph_no += 1
            last_node_in_graph += 1
            graph_attributes = args.node_attributes_file.readline().strip()
            graph_attributes += ', ' + one_hot(args.node_labels_file.readline().strip()) + '\n'
            graph_edges = first_edge_in_next_graph
    
    # Finalize last graph
    edges_in_graph, first_edge_in_next_graph = get_edges(
        args, last_node_in_graph)
    graph_edges += edges_in_graph

    write_graph(args.output_dir, graph_no, graph_attributes, graph_edges)


def main():
    args = parser()
    extract_graphs(args)
    args.edge_table_file.close()
    args.graph_indicator_file.close()
    args.node_attributes_file.close()
    args.node_labels_file.close()

if __name__ == '__main__':
    main()

