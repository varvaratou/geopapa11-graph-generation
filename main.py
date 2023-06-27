import multiprocessing
import os
import random
import shutil
import torch
from datetime import datetime
from time import localtime, strftime
from random import shuffle
from tensorboard_logger import configure

import create_graphs
from build_dataset import construct_graphs
from args import Args
from data import GraphSequenceSamplerPytorchNobfs, GraphSequenceSamplerPytorchCanonical, GraphSequenceSamplerPytorch
from model import GRUPlain, MLPVAEConditionalPlain, MLPPlain, LSTMPlain
from train import train, train_graph_completion
from utils import draw_graph_list, save_graph_list

from data import test_encode_decode_adj

# todo geopapa: try also other types of graphs  -- [04/19/2023] I tried and it worked!
# todo geopapa: make sure it's also working for GraphRNN_RNN -- [04/19/2023] I made the changes and it worked!


"""This is the main script where everything is called.

author: geopapa (07/25/2022)
"""
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # All necessary arguments are defined in args.py
    args = Args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    # print('CUDA:', args.cuda)
    print('File name prefix:', args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y%m%d_%H%M%S", localtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir('tensorboard'):
            shutil.rmtree('tensorboard')
    configure(f'tensorboard/run_{time}', flush_secs=5)

    # if you use pre-saved graphs
    # dir_input = '/dfs/scratch0/jiaxuany0/graphs/
    # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    if args.graph_type == 'dynamo':
        graphs, num_node_labels = construct_graphs(args)
    else:
        graphs = create_graphs.create(args)

    graphs = graphs[:12]

    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(args.train_size_frac * graphs_len):]
    graphs_train = graphs[0:int(args.train_size_frac * graphs_len)]
    graphs_validate = graphs[0:int(args.valid_size_frac * graphs_len)]

    timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
    draw_graph_list(graphs_train[:9], 3, 3, fname=f'figures/train')
    draw_graph_list(graphs_test[:3], 1, 3, fname=f'figures/test')
    draw_graph_list(graphs_validate[:2], 1, 2, fname=f'figures/validate')

    graph_validate_len = 0  # average number of nodes in `graphs_validate`
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print('Avg number of nodes in `graphs_validate`:', graph_validate_len)

    graph_test_len = 0  # average number of nodes in `graph_test_len`
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('Avg number of nodes in `graph_test_len`:', graph_test_len)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])  # max number of nodes in a graph
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])  # max number of edges in a graph
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])  # min number of edges in a graph

    # args.max_num_node = 2000
    # show graph statistics
    print(f'Number of graphs: {len(graphs)}, Number of graphs in training set: {len(graphs_train)}')
    print(f'max number of nodes: {args.max_num_node}')
    print(f'min/max number of edges: {min_num_edge}; {max_num_edge}')
    print(f'max previous nodes: {args.max_prev_node}')

    # save ground truth graphs (to get train and test set, after loading, you need to manually slice)
    save_graph_list(graphs, os.path.join(args.graph_save_path, args.fname_train + 'epoch_0.dat'))
    save_graph_list(graphs, os.path.join(args.graph_save_path, args.fname_test + 'epoch_0.dat'))
    print('train and test graphs saved at: ', args.graph_save_path)

    # comment when normal training, for graph completion only
    # p = 0.5
    # for graph in graphs_train:
    #     for node in list(graph.nodes()):
    #         # print('node', node)
    #         if np.random.rand() > p:
    #             graph.remove_node(node)
    #     for edge in list(graph.edges()):
    #         # print('edge', edge)
    #         if np.random.rand() > p:
    #             graph.remove_edge(edge[0], edge[1])
    test_encode_decode_adj()

    # dataset initialization
    if 'nobfs' in args.note:
        print('nobfs')
        dataset = GraphSequenceSamplerPytorchNobfs(graphs_train, max_num_node=args.max_num_node)
        args.max_prev_node = args.max_num_node - 1
    if 'barabasi_noise' in args.graph_type:
        print('barabasi_noise')
        dataset = GraphSequenceSamplerPytorchCanonical(graphs_train, max_prev_node=args.max_prev_node)
        args.max_prev_node = args.max_num_node - 1
    else:
        dataset = GraphSequenceSamplerPytorch(graphs_train, max_num_node=args.max_num_node, max_prev_node=args.max_prev_node)

    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset) for i in range(len(dataset))], num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, sampler=sample_strategy)

    # model initialization
    # Graph RNN VAE model
    # lstm = LSTMPlain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                  hidden_size=args.hidden_size, num_layers=args.num_layers, device=args.device)

    # rnn: f_trans (graph-level RNN), output: f_out (theta_i, edge-level model)
    if args.note.startswith('GraphRNN_VAE'):
        rnn = GRUPlain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            transform_input=True,
            device=args.device
        )
        output = MLPVAEConditionalPlain(
            h_size=args.hidden_size_rnn,
            embedding_size=args.embedding_size_output,
            y_size=args.max_prev_node,
            device=args.device
        )

    elif args.note == 'GraphRNN_MLP':  # GraphRNN-S (theta_i \in R^{i-1}: theta_i[j]: probability of edge (i, j))
        # We add +1 to account for nodes which have not been assigned a label yet (that's label value 0)
        vocab_size_node_label = None if args.vocab_size_node_label is None else args.vocab_size_node_label + 1

        rnn = GRUPlain(
            input_size=args.max_prev_node if args.vocab_size_node_label is None else args.max_num_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            transform_input=True,
            vocab_size_node_label=vocab_size_node_label,
            embedding_size_node_label=args.embedding_size_node_label,
            device=args.device
        )

        if args.vocab_size_node_label is None:
            output = MLPPlain(
                h_size=args.hidden_size_rnn,
                embedding_size=args.embedding_size_output,
                y_size=args.max_prev_node,
            )
        else:
            output = {
                'graph_structure': MLPPlain(
                    h_size=args.hidden_size_rnn,
                    embedding_size=args.embedding_size_rnn,
                    y_size=args.max_num_node,
                ),
                'node_labels': MLPPlain(
                    h_size=args.hidden_size_rnn,
                    embedding_size=args.embedding_size_rnn,
                    y_size=vocab_size_node_label,
                )
            }

    elif args.note == 'GraphRNN_RNN':
        # We add +1 to account for nodes which have not been assigned a label yet (that's label value 0)
        vocab_size_node_label = None if args.vocab_size_node_label is None else args.vocab_size_node_label + 1

        rnn = GRUPlain(
            input_size=args.max_prev_node if args.vocab_size_node_label is None else args.max_num_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            transform_input=True,
            output_size=args.hidden_size_rnn_output,
            vocab_size_node_label=vocab_size_node_label,
            embedding_size_node_label=args.embedding_size_node_label,
            device=args.device
        )

        if args.vocab_size_node_label is None:
            output = GRUPlain(
                input_size=1,
                embedding_size=args.embedding_size_rnn_output,
                hidden_size=args.hidden_size_rnn_output,
                num_layers=args.num_layers,
                transform_input=True,
                output_size=1,
                device=args.device
            )
        else:
            output = {
                'graph_structure': GRUPlain(
                    input_size=1,
                    embedding_size=args.embedding_size_rnn_output,
                    hidden_size=args.hidden_size_rnn_output,
                    num_layers=args.num_layers,
                    transform_input=True,
                    output_size=1,
                    device=args.device
                ),
                'node_labels': MLPPlain(
                    h_size=rnn.output_size if rnn.output_size is not None else rnn.hidden_size,
                    embedding_size=args.embedding_size_rnn,
                    y_size=vocab_size_node_label,
                )
            }

    rnn = rnn.to(args.device)
    if isinstance(output, dict):
        for key in output:
            output[key] = output[key].to(args.device)
    else:
        output = output.to(args.device)

    # just printing RNN and output model information
    print(rnn)
    if isinstance(output, dict):
        for key in output:
            print(output[key])
    else:
        print(output)

    # train
    train(args, dataset_loader, rnn, output, device=args.device)

    # graph completion
    train_graph_completion(args, dataset_loader, rnn, output, device=args.device)

    # nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter=200,
    #           graph_validate_len=graph_validate_len, graph_test_len=graph_test_len, device=args.device)
