import networkx as nx
import numpy as np

def construct_graphs(args):
    graphs = []
    num_node_labels = 0
    if args.graph_type == 'dynamo':
        graphs, num_node_labels = load_graph_batch(
            min_num_nodes=4, max_num_nodes=1000, name='DYNAMO', node_labels=True, graph_labels=False)
        args.max_prev_node = 10

    return graphs, num_node_labels


def load_graph_batch(min_num_nodes, max_num_nodes, name, node_labels, graph_labels):
    print('Loading graph dataset: ' + str(name))
    path = 'dataset/' + name + '/'

    # Load data from text files
    data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(
        path + name + '_graph_indicator.txt', delimiter=',').astype(int)
    num_node_labels = None
    if node_labels:
        data_node_label = np.loadtxt(
            path + name + '_node_labels.txt', delimiter=',').astype(int)
        num_node_labels = len(np.unique(data_node_label))
        one_hot_node_labels = (np.arange(num_node_labels)
                               == data_node_label[:, None]).astype(np.float32)
    if graph_labels:
        data_graph_labels = np.loadtxt(
            path + name + '_graph_labels.txt', delimiter=',').astype(int)
    data_tuple = list(map(tuple, data_adj))

    # Generate graph objects from data
    G = nx.Graph()
    # Add edges
    G.add_edges_from(data_tuple)
    # Add node labels
    if node_labels:
        #for i in range(data_node_label.shape[0]):
        #    # Add the one hot labels
        #    G.add_node(i+1, label=data_node_label[i])
        #    G.add_node(i+1, one_hot_label=one_hot_node_labels[i])
        keys = list(range(1, len(data_node_label)+1))
        nodes_with_labels = dict(zip(keys, data_node_label))
        G._node.update(nodes_with_labels)

    G.remove_nodes_from(list(nx.isolates(G)))

    # Split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # Find the nodes for each graph
        nodes = node_list[data_graph_indicator == i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]

        if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()

    print('Graphs loaded, total num: {}'.format(len(graphs)))
    return graphs, num_node_labels
