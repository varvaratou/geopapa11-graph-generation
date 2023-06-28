import networkx as nx
import numpy as np
import os
import time as tm
import torch
import torch.nn.functional as F

from time import localtime, strftime
from tensorboard_logger import log_value
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import binary_cross_entropy_weight, sample_sigmoid, sample_softmax, sample_sigmoid_supervised, \
    sample_softmax_supervised, sample_sigmoid_supervised_simple, sample_softmax_supervised_simple
from data import decode_adj
from utils import get_graph, get_graph_with_labels, draw_graph_list, save_graph_list


# from utils import *
# from model import *
# from data import *


# import create_graphs


def train_vae_epoch(epoch, args, rnn, output, data_loader, optim_rnn, optim_output, scheduler_rnn, scheduler_output, device='cpu'):
    rnn.train()
    if isinstance(output, dict):
        for key in output:
            output[key].train()
    else:
        output.train()

    num_batches_in_epoch = len(data_loader)
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        if isinstance(output, dict):
            for key in output:
                output[key].zero_grad()
        else:
            output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # output.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = Variable(x).to(device)
        y = Variable(y).to(device)

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        y_pred, z_mu, z_lsgms = output(h)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
        z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
        z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
        z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
        # use cross entropy loss
        loss_bce = binary_cross_entropy_weight(y_pred, y, device=args.device)
        loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
        loss_kl /= y.size(0)*y.size(1)*sum(y_len)  # normalize
        loss = loss_bce + loss_kl
        loss.backward()

        # update deterministic output and LSTM
        if isinstance(optim_output, dict):
            for key in optim_output:
                optim_output[key].step()
        else:
            optim_output.step()
        optim_rnn.step()

        if isinstance(scheduler_output, dict):
            for key in scheduler_output:
                scheduler_output[key].step()
        else:
            scheduler_output.step()
        scheduler_rnn.step()

        z_mu_mean = torch.mean(z_mu.data)
        z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
        z_mu_min = torch.min(z_mu.data)
        z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
        z_mu_max = torch.max(z_mu.data)
        z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)

        # only output first or last batch's statistics
        if (epoch == 1 or epoch % args.epochs_log == 0) and (batch_idx == 0 or batch_idx == len(data_loader) - 1):
            print(f'[{strftime("%Y-%m-%d %H:%M:%S", localtime())}] '
                  f'Epoch: {epoch}/{args.epochs}, batch: {str(batch_idx+1).rjust(len(str(num_batches_in_epoch)))}/{num_batches_in_epoch}: '
                  f'train bce loss: {loss_bce.item():.6f}, train kl loss: {loss_kl.item():.6f}, '
                  f'graph type: {args.graph_type}, num_layers: {args.num_layers}, hidden: {args.hidden_size_rnn}')
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean,
                  'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_' + args.fname, loss_bce.item(), epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' + args.fname, loss_kl.item(), epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_' + args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_' + args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_' + args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_' + args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_' + args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_' + args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)

        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)


def test_vae_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1, device='cpu'):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    if isinstance(output, dict):
        for key in output:
            output[key].eval()
    else:
        output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # discrete prediction
    x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(device)
    # y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # normalized prediction score
    # y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # discrete prediction
    # x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()

    for i in range(max_num_node):
        h = rnn(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True, sample_time=sample_time, device=device)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = Variable(rnn.hidden.data).to(device)
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)

    return G_pred_list


def test_vae_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1, device='cpu'):
    """Algorith 1 of the paper."""
    rnn.eval()
    if isinstance(output, dict):
        for key in output:
            output[key].eval()
    else:
        output.eval()

    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # discrete prediction
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(device)
        # y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # normalized prediction score
        # y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # discrete prediction
        # x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()

        for i in range(max_num_node):
            # print('finish node:', i)
            h = rnn(x_step)
            y_pred_step, _, _ = output(h)

            if args.vocab_size_node_label is None:
                y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
                x_step = sample_sigmoid_supervised(
                    y_pred_step,
                    y[:, i:i + 1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )
            else:
                y_pred[:, i:i + 1, :] = F.softmax(y_pred_step)
                x_step = sample_softmax_supervised(
                    y_pred_step,
                    y[:, i:i + 1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )

            y_pred_long[:, i:i + 1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_epoch(epoch, args, rnn, output, data_loader, optim_rnn, optim_output, scheduler_rnn, scheduler_output, device='cpu'):
    rnn.train()
    if isinstance(output, dict):
        for key in output:
            output[key].train()
    else:
        output.train()

    num_batches_in_epoch = len(data_loader)
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        if isinstance(output, dict):
            for key in output:
                output[key].zero_grad()
        else:
            output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']

        y_len_max = max(y_len_unsorted)  # get the largest graph in the batch
        x_unsorted = x_unsorted[:, 0:y_len_max]
        y_unsorted = y_unsorted[:, 0:y_len_max]

        # rnn.hidden= h^{<0>} (initialize LSTM hidden state according to batch size)
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input based on graph size
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = Variable(torch.index_select(x_unsorted, 0, sort_index)).to(device)
        y = Variable(torch.index_select(y_unsorted, 0, sort_index)).to(device)

        if args.vocab_size_node_label is not None:
            x_node_labels_unsorted = data['x_node_labels']
            y_node_labels_unsorted = data['y_node_labels']
            x_node_labels_unsorted = x_node_labels_unsorted[:, 0:y_len_max]
            y_node_labels_unsorted = y_node_labels_unsorted[:, 0:y_len_max]
            x_node_labels = Variable(torch.index_select(x_node_labels_unsorted, 0, sort_index)).to(device=device)
            y_node_labels = Variable(torch.index_select(y_node_labels_unsorted, 0, sort_index)).to(device=device, dtype=torch.long)

            if not isinstance(output, dict):
                raise ValueError('We need two output heads. One for graph structure and another for node labels.')

            h = rnn((x, x_node_labels), pack=True, input_len=y_len)

            y_pred_node_labels_logits = output['node_labels'](h)  # node label logits
            y_pred_graph_logits = output['graph_structure'](h)  # graph logits

            y_pred_graph_prob = torch.sigmoid(y_pred_graph_logits)

            # clean predictions (we take into account predictions up to `y_len` per example)
            y_pred_graph_prob = pack_padded_sequence(y_pred_graph_prob, y_len, batch_first=True)
            y_pred_graph_prob = pad_packed_sequence(y_pred_graph_prob, batch_first=True)[0]

            # clean predictions (we take into account predictions up to `y_len` per example)
            y_pred_node_labels_logits = pack_padded_sequence(y_pred_node_labels_logits, y_len, batch_first=True)
            y_pred_node_labels_logits = pad_packed_sequence(y_pred_node_labels_logits, batch_first=True)[0]

            # loss of graph structure
            loss_graph = binary_cross_entropy_weight(y_pred_graph_prob, y, device=device)

            # loss between the predicted and true labels of current node
            loss_node_label = F.cross_entropy(y_pred_node_labels_logits.permute((0, 2, 1)), y_node_labels, reduction='none').to(device)
            boolean_mask = torch.ones_like(loss_node_label)
            boolean_mask = pack_padded_sequence(boolean_mask, y_len, batch_first=True)  # we need this as we have graphs of different
            boolean_mask = pad_packed_sequence(boolean_mask, batch_first=True)[0]  # size in a batch
            loss_node_label *= boolean_mask
            loss_node_label = loss_node_label.mean()

            loss = (4. * loss_graph + loss_node_label) / 5.
        else:
            h = rnn(x, pack=True, input_len=y_len)
            y_pred_graph_logits = output(h)  # logits
            y_pred_graph_prob = torch.sigmoid(y_pred_graph_logits)  # each prediction represents a probability this edge being 1

            # clean predictions (we take into account predictions up to `y_len` per example)
            y_pred_graph_prob = pack_padded_sequence(y_pred_graph_prob, y_len, batch_first=True)
            y_pred_graph_prob = pad_packed_sequence(y_pred_graph_prob, batch_first=True)[0]
            loss = binary_cross_entropy_weight(y_pred_graph_prob, y, device=device)

        # update gradients
        loss.backward()

        # update deterministic output and LSTM
        if isinstance(optim_output, dict):
            for key in optim_output:
                optim_output[key].step()
        else:
            optim_output.step()
        optim_rnn.step()

        if isinstance(scheduler_output, dict):
            for key in scheduler_output:
                scheduler_output[key].step()
        else:
            scheduler_output.step()
        scheduler_rnn.step()

        # only output first or last batch's statistics
        if (epoch == 1 or epoch % args.epochs_log == 0) and (batch_idx == 0 or batch_idx == len(data_loader)-1):
            print(f'[{strftime("%Y-%m-%d %H:%M:%S", localtime())}] '
                  f'Epoch: {epoch}/{args.epochs}, batch: {str(batch_idx+1).rjust(len(str(num_batches_in_epoch)))}/{num_batches_in_epoch}: '
                  f'train loss: {loss.item():.6f}, graph type: {args.graph_type}, num_layers: {args.num_layers}, '
                  f'hidden: {args.hidden_size_rnn}')

        # logging
        log_value('loss_' + args.fname, loss.item(), epoch*args.batch_ratio + batch_idx)
        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)


def test_mlp_epoch(epoch, args, rnn, output, test_batch_size=16, save_histogram=False, sample_time=1, device='cpu'):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    if isinstance(output, dict):
        for key in output:
            output[key].eval()
    else:
        output.eval()

    # generate graphs
    if args.vocab_size_node_label is None:
        y_pred_graph = Variable(torch.zeros(test_batch_size, args.max_num_node, args.max_prev_node)).to(device)  # normalized prediction
        y_pred_graph_long = Variable(torch.zeros(test_batch_size, args.max_num_node, args.max_prev_node)).to(device)  # discrete prediction
        x_graph_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(device)
    else:
        n = args.max_num_node
        y_pred_graph = Variable(torch.zeros(test_batch_size, n, n)).to(device)
        y_pred_graph_long = Variable(torch.zeros(test_batch_size, n, n)).to(device)
        y_pred_node_labels = Variable(torch.zeros(test_batch_size, n, args.vocab_size_node_label + 1)).to(device)
        y_pred_node_labels_long = Variable(torch.zeros(test_batch_size, n)).to(torch.int64).to(device)
        x_graph_step = Variable(torch.ones(test_batch_size, 1, n)).to(device)
        x_node_labels_step = Variable(torch.zeros(test_batch_size, 1, n)).to(torch.int64).to(device)

    for i in range(args.max_num_node):
        if args.vocab_size_node_label is None:
            h = rnn(x_graph_step)
            y_pred_graph_logits_step = output(h)
            y_pred_graph[:, i:i+1, :] = torch.sigmoid(y_pred_graph_logits_step)
            x_graph_step = sample_sigmoid(y_pred_graph_logits_step, sample=True, sample_time=sample_time, device=device)
            y_pred_graph_long[:, i:i+1, :] = x_graph_step
        else:
            if not isinstance(output, dict):
                raise ValueError('We need two output heads. One for graph structure and another for node labels.')

            h = rnn((x_graph_step, x_node_labels_step))
            y_pred_node_labels_logits_step = output['node_labels'](h)  # node label logits
            y_pred_graph_logits_step = output['graph_structure'](h)  # graph logits

            y_pred_graph[:, i:i+1, :] = torch.sigmoid(y_pred_graph_logits_step)
            y_pred_node_labels[:, i:i+1, :] = F.softmax(y_pred_node_labels_logits_step, dim=-1)

            x_graph_step = sample_sigmoid(y_pred_graph_logits_step, sample=True, sample_time=sample_time, device=device)
            x_graph_step[:, 0, i+1:] = 0.
            x_node_labels_step[:, 0, i:i+1] = sample_softmax(y_pred_node_labels_logits_step, sample=True, device=device)

            y_pred_graph_long[:, i:i+1, :] = x_graph_step
            y_pred_node_labels_long[:, i:i+1] = x_node_labels_step[:, 0, i:i+1]

        rnn.hidden = Variable(rnn.hidden.data).to(device)

    if args.vocab_size_node_label is not None:
        y_pred_graph_long[:, 0, 0] = 0.  # remove the edge of first node to itself (that was just a dummy edge)
        y_pred_node_labels = y_pred_node_labels.data
        y_pred_node_labels_long_data = y_pred_node_labels_long.data.long()

    y_pred_graph_data = y_pred_graph.data
    y_pred_graph_long_data = y_pred_graph_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        if args.vocab_size_node_label is None:
            adj_pred = decode_adj(y_pred_graph_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        else:
            adj = y_pred_graph_long_data[i].cpu().numpy()
            adj = adj + adj.T  # make adjacent matrix symmetric
            np.fill_diagonal(adj, 0)  # remove any self connections
            adj = (adj > 0).astype(int)
            node_labels = y_pred_node_labels_long_data[i].cpu().numpy()

            # remove nodes which don't connect to any other nodes
            boolean_mask = ~np.all(adj == 0, 1)
            non_zero_idx = np.where(boolean_mask)[0]
            adj = adj[np.ix_(non_zero_idx, non_zero_idx)]
            node_labels = node_labels[non_zero_idx]

            G_pred = nx.from_numpy_matrix(adj)
            G_pred.node = dict(zip(G_pred.nodes(), node_labels))
        G_pred_list.append(G_pred)

    # # save prediction histograms, plot histogram over each time step
    # if save_histogram:
    #     save_prediction_histogram(y_pred_data.cpu().numpy(),
    #                           fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
    #                           max_num_node=max_num_node)
    return G_pred_list


def test_mlp_partial_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1, device='cpu'):
    rnn.eval()
    if isinstance(output, dict):
        for key in output:
            output[key].eval()
    else:
        output.eval()
    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = x.size(0)

        # rnn.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # discrete prediction
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(device)
        # y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # normalized prediction score
        # y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # discrete prediction
        # x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()

        for i in range(max_num_node):
            # print('finish node:', i)
            h = rnn(x_step)
            y_pred_step = output(h)

            if args.vocab_size_node_label is None:
                y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
                x_step = sample_sigmoid_supervised(
                    y_pred_step,
                    y[:, i:i + 1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )
            else:
                y_pred[:, i:i + 1, :] = F.softmax(y_pred_step)
                x_step = sample_softmax_supervised(
                    y_pred_step,
                    y[:, i:i + 1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )

            y_pred_long[:, i:i+1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_mlp_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1, device='cpu'):
    """Algorith 1 of the paper."""
    rnn.eval()
    if isinstance(output, dict):
        for key in output:
            output[key].eval()
    else:
        output.eval()

    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = y.size(0)

        # rnn.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
        rnn.hidden = rnn.init_hidden(test_batch_size)
        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # discrete prediction
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(device)
        # y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # normalized prediction score
        # y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda()  # discrete prediction
        # x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).cuda()

        for i in range(max_num_node):
            # print('finish node:', i)
            h = rnn(x_step)
            y_pred_step = output(h)

            if args.vocab_size_node_label is None:
                y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
                x_step = sample_sigmoid_supervised_simple(
                    y_pred_step,
                    y[:, i:i+1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )
            else:
                y_pred[:, i:i + 1, :] = F.softmax(y_pred_step)
                x_step = sample_softmax_supervised_simple(
                    y_pred_step,
                    y[:, i:i+1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )

            y_pred_long[:, i:i+1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def test_rnn_partial_simple_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1, device='cpu'):
    """Algorith 1 of the paper."""
    rnn.eval()
    if isinstance(output, dict):
        for key in output:
            output[key].eval()
    else:
        output.eval()

    G_pred_list = []
    for batch_idx, data in enumerate(data_loader):
        x = data['x'].float()
        y = data['y'].float()
        y_len = data['len']
        test_batch_size = y.size(0)  # test_batch_size = x.size(0)

        # rnn.hidden= h^{<0>} (initialize LSTM hidden state according to batch size)
        rnn.hidden = rnn.init_hidden(test_batch_size)

        # generate graphs
        max_num_node = int(args.max_num_node)
        y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # normalized prediction score
        y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).to(device)  # discrete prediction
        x_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(device)

        for i in range(max_num_node):
            h = rnn(x_step)
            hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(2))).to(device)
            output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)  # num_layers, batch_size, hidden_size
            output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).to(device)
            y_pred_step = output(output_x_step)

            if args.vocab_size_node_label is None:
                y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
                x_step = sample_sigmoid_supervised_simple(
                    y_pred_step,
                    y[:, i:i + 1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )
            else:
                y_pred[:, i:i + 1, :] = F.softmax(y_pred_step)
                x_step = sample_softmax_supervised_simple(
                    y_pred_step,
                    y[:, i:i + 1, :].to(device),
                    current=i,
                    y_len=y_len,
                    sample_time=sample_time,
                    device=device
                )

            y_pred_long[:, i:i+1, :] = x_step
            rnn.hidden = Variable(rnn.hidden.data).to(device)
        y_pred_data = y_pred.data
        y_pred_long_data = y_pred_long.data.long()

        # save graphs as pickle
        for i in range(test_batch_size):
            adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
            G_pred_list.append(G_pred)
    return G_pred_list


def train_mlp_forward_epoch(epoch, args, rnn, output, data_loader, device='cpu'):
    rnn.train()
    if isinstance(output, dict):
        for key in output:
            output[key].train()
    else:
        output.train()

    num_batches_in_epoch = len(data_loader)
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        if isinstance(output, dict):
            for key in output:
                output[key].zero_grad()
        else:
            output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # rnn.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)
        x = Variable(x).to(device)
        y = Variable(y).to(device)

        h = rnn(x, pack=True, input_len=y_len)
        y_pred = output(h)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        # use cross entropy loss

        loss = 0
        for j in range(y.size(1)):
            # print('y_pred',y_pred[0,j,:],'y',y[0,j,:])
            end_idx = min(j+1, y.size(2))
            loss += binary_cross_entropy_weight(
                y_pred[:, j, 0:end_idx],
                y[:, j, 0:end_idx],
                device=device
            ) * end_idx

        # only output first or last batch's statistics
        if (epoch == 1 or epoch % args.epochs_log == 0) and (batch_idx == 0 or batch_idx == len(data_loader) - 1):
            print(f'[{strftime("%Y-%m-%d %H:%M:%S", localtime())}] '
                  f'Epoch: {epoch}/{args.epochs}, batch: {str(batch_idx+1).rjust(len(str(num_batches_in_epoch)))}/{num_batches_in_epoch}: '
                  f'train loss: {loss.item():.6f}, graph type: {args.graph_type}, num_layers: {args.num_layers}, '
                  f'hidden: {args.hidden_size_rnn}')

        # logging
        log_value('loss_' + args.fname, loss.item(), epoch*args.batch_ratio+batch_idx)

        loss_sum += loss.item()
    return loss_sum/(batch_idx+1)


# too complicated, deprecated
# def test_mlp_partial_bfs_epoch(epoch, args, rnn, output, data_loader, save_histogram=False, sample_time=1, device='cpu'):
#     rnn.eval()
#     if isinstance(output, dict):
#         for key in output:
#             output[key].eval()
#     else:
#         output.eval()
#     G_pred_list = []
#     for batch_idx, data in enumerate(data_loader):
#         x = data['x'].float()
#         y = data['y'].float()
#         y_len = data['len']
#         test_batch_size = x.size(0)
#         rnn.hidden = rnn.init_hidden(test_batch_size)
#         # generate graphs
#         max_num_node = int(args.max_num_node)
#         y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # normalized prediction score
#         y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda() # discrete prediction
#         x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda()
#         for i in range(max_num_node):
#             # 1 back up hidden state
#             hidden_prev = Variable(rnn.hidden.data).cuda()
#             h = rnn(x_step)
#             y_pred_step = output(h)
#             y_pred[:, i:i + 1, :] = torch.sigmoid(y_pred_step)
#             x_step = sample_sigmoid_supervised(y_pred_step, y[:,i:i+1,:].cuda(), current=i, y_len=y_len, sample_time=sample_time)
#             y_pred_long[:, i:i + 1, :] = x_step
#
#             rnn.hidden = Variable(rnn.hidden.data).cuda()
#
#             # print('finish node:', i)
#         y_pred_data = y_pred.data
#         y_pred_long_data = y_pred_long.data.long()
#
#         # save graphs as pickle
#         for i in range(test_batch_size):
#             adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
#             G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
#             G_pred_list.append(G_pred)
#     return G_pred_list


def train_rnn_epoch(epoch, args, rnn, output, data_loader, optim_rnn, optim_output, scheduler_rnn, scheduler_output, device='cpu'):
    rnn.train()
    if isinstance(output, dict):
        for key in output:
            output[key].train()
    else:
        output.train()

    num_batches_in_epoch = len(data_loader)
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        if isinstance(output, dict):
            for key in output:
                output[key].zero_grad()
        else:
            output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']

        y_len_max = max(y_len_unsorted)  # get the largest graph in the batch
        x_unsorted = x_unsorted[:, 0:y_len_max]
        y_unsorted = y_unsorted[:, 0:y_len_max]

        # rnn.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input based on graph size
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index).to(device)
        y = torch.index_select(y_unsorted, 0, sort_index).to(device)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1, b2_l1, ..., b1_l2, b2_l2, ...
        # y_reshape[t*batch_size:(t+1)*batch_size] = y[:batch_size, t]
        # that is, the adj matrix of the t-th node (t-th time step) is in the contiguous portion t*batch_size:(t+1)*batch_size of y_reshape
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = torch.LongTensor([i for i in range(y_reshape.size(0)-1, -1, -1)])
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])  # count how many y_len is greater or equal i
            output_y_len.extend([min(i, y.size(2))]*count_temp)  # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(device)
        y = Variable(y).to(device)
        output_x = Variable(output_x).to(device)
        output_y = Variable(output_y).to(device)

        # if using ground truth to train
        if args.vocab_size_node_label is not None:
            x_node_labels_unsorted = data['x_node_labels']
            y_node_labels_unsorted = data['y_node_labels']
            x_node_labels_unsorted = x_node_labels_unsorted[:, 0:y_len_max]
            y_node_labels_unsorted = y_node_labels_unsorted[:, 0:y_len_max]
            x_node_labels = Variable(torch.index_select(x_node_labels_unsorted, 0, sort_index)).to(device)
            y_node_labels = Variable(torch.index_select(y_node_labels_unsorted, 0, sort_index)).to(device=device, dtype=torch.long)

            if not isinstance(output, dict):
                raise ValueError('We need two output heads. One for graph structure and another for node labels.')

            h = rnn((x, x_node_labels), pack=True, input_len=y_len)

            # node label logits
            y_pred_node_labels_logits = output['node_labels'](h)
            # clean predictions (we take into account predictions up to `y_len` per example)
            y_pred_node_labels_logits = pack_padded_sequence(y_pred_node_labels_logits, y_len, batch_first=True)
            y_pred_node_labels_logits = pad_packed_sequence(y_pred_node_labels_logits, batch_first=True)[0]

            # loss of graph structure
            h = pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
            idx = Variable(torch.LongTensor([i for i in range(h.size(0)-1, -1, -1)])).to(device)  # reverse h
            h = h.index_select(0, idx)
            hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).to(device)
            # output.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
            output['graph_structure'].hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
            y_pred_graph_logits = output['graph_structure'](output_x, pack=True, input_len=output_y_len)  # graph logits
            y_pred_graph_prob = torch.sigmoid(y_pred_graph_logits)

            # clean predictions (we take into account predictions up to `y_len` per example)
            y_pred_graph_prob = pack_padded_sequence(y_pred_graph_prob, output_y_len, batch_first=True)
            y_pred_graph_prob = pad_packed_sequence(y_pred_graph_prob, batch_first=True)[0]
            output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
            output_y = pad_packed_sequence(output_y, batch_first=True)[0]

            loss_graph = binary_cross_entropy_weight(y_pred_graph_prob, output_y, device=device)

            # loss between the predicted and true labels of current node
            loss_node_label = F.cross_entropy(y_pred_node_labels_logits.permute((0, 2, 1)), y_node_labels, reduction='none').to(device)
            boolean_mask = torch.ones_like(loss_node_label)
            boolean_mask = pack_padded_sequence(boolean_mask, y_len, batch_first=True)  # we need this as we have graphs of different
            boolean_mask = pad_packed_sequence(boolean_mask, batch_first=True)[0]  # size in a batch
            loss_node_label *= boolean_mask
            loss_node_label = loss_node_label.mean()

            loss = (4. * loss_graph + loss_node_label) / 5.
        else:
            h = rnn(x, pack=True, input_len=y_len)
            h = pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
            idx = Variable(torch.LongTensor([i for i in range(h.size(0)-1, -1, -1)])).to(device)  # reverse h
            h = h.index_select(0, idx)
            hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).to(device)
            # output.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
            output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
            y_pred_graph_logits = output(output_x, pack=True, input_len=output_y_len)
            y_pred_graph_prob = torch.sigmoid(y_pred_graph_logits)
            # clean predictions (we take into account predictions up to `y_len` per example)
            y_pred_graph_prob = pack_padded_sequence(y_pred_graph_prob, output_y_len, batch_first=True)
            y_pred_graph_prob = pad_packed_sequence(y_pred_graph_prob, batch_first=True)[0]
            output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
            output_y = pad_packed_sequence(output_y, batch_first=True)[0]

            # use cross entropy loss
            loss = binary_cross_entropy_weight(y_pred_graph_prob, output_y, device=device)

        # update gradients
        loss.backward()

        # update deterministic output and LSTM
        if isinstance(optim_output, dict):
            for key in optim_output:
                optim_output[key].step()
        else:
            optim_output.step()
        optim_rnn.step()

        if isinstance(scheduler_output, dict):
            for key in scheduler_output:
                scheduler_output[key].step()
        else:
            scheduler_output.step()
        scheduler_rnn.step()

        # only output first or last batch's statistics
        if (epoch == 1 or epoch % args.epochs_log == 0) and (batch_idx == 0 or batch_idx == len(data_loader) - 1):
            print(f'[{strftime("%Y-%m-%d %H:%M:%S", localtime())}] '
                  f'Epoch: {epoch}/{args.epochs}, batch: {str(batch_idx+1).rjust(len(str(num_batches_in_epoch)))}/{num_batches_in_epoch}: '
                  f'train loss: {loss.item():.6f}, graph type: {args.graph_type}, num_layers: {args.num_layers}, '
                  f'hidden: {args.hidden_size_rnn}')

        # logging
        log_value('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)
        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim
    return loss_sum/(batch_idx+1)


def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16, device='cpu'):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    if isinstance(output, dict):
        for key in output:
            output[key].eval()
    else:
        output.eval()

    # generate graphs
    if args.vocab_size_node_label is None:
        y_pred_graph_long = Variable(torch.zeros(test_batch_size, args.max_num_node, args.max_prev_node)).to(device)  # discrete prediction
        x_graph_step = Variable(torch.ones(test_batch_size, 1, args.max_prev_node)).to(device)
    else:
        n = args.max_num_node
        y_pred_graph_long = Variable(torch.zeros(test_batch_size, n, n)).to(device)
        y_pred_node_labels_long = Variable(torch.zeros(test_batch_size, n)).to(torch.int64).to(device)
        x_graph_step = Variable(torch.ones(test_batch_size, 1, n)).to(device)
        x_node_labels_step = Variable(torch.zeros(test_batch_size, 1, n)).to(torch.int64).to(device)

    for i in range(args.max_num_node):
        if args.vocab_size_node_label is None:
            h = rnn(x_graph_step)
            hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(2))).to(device)
            # output.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
            output.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)

            output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).to(device)
            for j in range(min(args.max_prev_node, i+1)):
                output_y_pred_step = output(output_x_step)
                output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1, device=device)
                x_graph_step[:, :, j:j+1] = output_x_step
                output.hidden = Variable(output.hidden.data).to(device)
            y_pred_graph_long[:, i:i+1, :] = x_graph_step
        else:
            if not isinstance(output, dict):
                raise ValueError('We need two output heads. One for graph structure and another for node labels.')

            h = rnn((x_graph_step, x_node_labels_step))
            hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(2))).to(device)
            # output.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
            output['graph_structure'].hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)

            # Sample graph structure
            output_x_step = Variable(torch.ones(test_batch_size, 1, 1)).to(device)
            for j in range(min(args.max_num_node, i+1)):
                output_y_pred_step = output['graph_structure'](output_x_step)
                output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1, device=device)
                x_graph_step[:, :, j:j+1] = output_x_step
                output['graph_structure'].hidden = Variable(output['graph_structure'].hidden.data).to(device)
            x_graph_step[:, 0, i+1:] = 0.
            y_pred_graph_long[:, i:i+1, :] = x_graph_step

            # Sample node labels
            y_pred_node_labels_logits_step = output['node_labels'](h)  # node label logits
            x_node_labels_step[:, 0, i:i+1] = sample_softmax(y_pred_node_labels_logits_step, sample=True, device=device)
            y_pred_node_labels_long[:, i:i+1] = x_node_labels_step[:, 0, i:i+1]

        rnn.hidden = Variable(rnn.hidden.data).to(device)

        #
        #     y_pred_graph[:, i:i+1, :] = torch.sigmoid(y_pred_graph_logits_step)
        #     y_pred_node_labels[:, i:i+1, :] = F.softmax(y_pred_node_labels_logits_step, dim=-1)
        #
        #     x_graph_step = sample_sigmoid(y_pred_graph_logits_step, sample=True, sample_time=sample_time, device=device)
        #     x_graph_step[:, 0, i+1:] = 0.
        #     x_node_labels_step[:, 0, i:i+1] = sample_softmax(y_pred_node_labels_logits_step, sample=True, device=device)
        #
        #     y_pred_graph_long[:, i:i+1, :] = x_graph_step
        #     y_pred_node_labels_long[:, i:i+1] = x_node_labels_step[:, 0, i:i+1]
        #
        # rnn.hidden = Variable(rnn.hidden.data).to(device)

    if args.vocab_size_node_label is not None:
        y_pred_graph_long[:, 0, 0] = 0.  # remove the edge of first node to itself (that was just a dummy edge)
        y_pred_node_labels_long_data = y_pred_node_labels_long.data.long()

    y_pred_graph_long_data = y_pred_graph_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        if args.vocab_size_node_label is None:
            adj_pred = decode_adj(y_pred_graph_long_data[i].cpu().numpy())
            G_pred = get_graph(adj_pred)  # get a graph from zero-padded adj
        else:
            adj = y_pred_graph_long_data[i].cpu().numpy()
            adj = adj + adj.T  # make adjacent matrix symmetric
            np.fill_diagonal(adj, 0)  # remove any self connections
            adj = (adj > 0).astype(int)
            node_labels = y_pred_node_labels_long_data[i].cpu().numpy()

            # remove nodes which don't connect to any other nodes
            boolean_mask = ~np.all(adj == 0, 1)
            non_zero_idx = np.where(boolean_mask)[0]
            adj = adj[np.ix_(non_zero_idx, non_zero_idx)]
            node_labels = node_labels[non_zero_idx]

            G_pred = nx.from_numpy_matrix(adj)
            G_pred.node = dict(zip(G_pred.nodes(), node_labels))
        G_pred_list.append(G_pred)

    return G_pred_list


def train_rnn_forward_epoch(epoch, args, rnn, output, data_loader, device='cpu'):
    rnn.train()
    if isinstance(output, dict):
        for key in output:
            output[key].train()
    else:
        output.train()

    num_batches_in_epoch = len(data_loader)
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        rnn.zero_grad()
        if isinstance(output, dict):
            for key in output:
                output[key].zero_grad()
        else:
            output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # rnn.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
        # output.hidden = output.init_hidden(batch_size=x_unsorted.size(0)*x_unsorted.size(1))

        # sort input
        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted, 0, sort_index)
        y = torch.index_select(y_unsorted, 0, sort_index)

        # input, output for output rnn module
        # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
        y_reshape = pack_padded_sequence(y, y_len, batch_first=True).data
        # reverse y_reshape, so that their lengths are sorted, add dimension
        idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
        idx = torch.LongTensor(idx)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, 1), y_reshape[:, 0:-1, 0:1]), dim=1)
        output_y = y_reshape
        # batch size for output module: sum(y_len)
        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin)-1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])  # count how many y_len is above i
            output_y_len.extend([min(i, y.size(2))]*count_temp)  # put them in output_y_len; max value should not exceed y.size(2)
        # pack into variable
        x = Variable(x).to(device)
        y = Variable(y).to(device)
        output_x = Variable(output_x).to(device)
        output_y = Variable(output_y).to(device)
        # print(output_y_len)
        # print('len', len(output_y_len))
        # print('y', y.size())
        # print('output_y', output_y.size())

        # if using ground truth to train
        h = rnn(x, pack=True, input_len=y_len)
        h = pack_padded_sequence(h, y_len, batch_first=True).data  # get packed hidden vector
        # reverse h
        idx = [i for i in range(h.size(0) - 1, -1, -1)]
        idx = Variable(torch.LongTensor(idx)).to(device)

        h = h.index_select(0, idx)
        hidden_null = Variable(torch.zeros(args.num_layers - 1, h.size(0), h.size(1))).to(device)
        # output.hidden= h^{<0>} (initialize LSTM hidden state according to batch size) (num_layers, batch_size, hidden_size)
        output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)
        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = torch.sigmoid(y_pred)
        # clean
        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]
        # use cross entropy loss
        loss = binary_cross_entropy_weight(y_pred, output_y, device=device)

        # only output first or last batch's statistics
        if (epoch == 1 or epoch % args.epochs_log == 0) and (batch_idx == 0 or batch_idx == len(data_loader) - 1):
            print(f'[{strftime("%Y-%m-%d %H:%M:%S", localtime())}] '
                  f'Epoch: {epoch}/{args.epochs}, batch: {str(batch_idx+1).rjust(len(str(num_batches_in_epoch)))}/{num_batches_in_epoch}: '
                  f'train loss: {loss.item():.6f}, graph type: {args.graph_type}, num_layers: {args.num_layers}, '
                  f'hidden: {args.hidden_size_rnn}')

        # logging
        log_value('loss_' + args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
        # print(y_pred.size())
        feature_dim = y_pred.size(0)*y_pred.size(1)
        loss_sum += loss.data[0]*feature_dim/y.size(0)
    return loss_sum/(batch_idx+1)


# train function for LSTM + VAE
def train(args, data_loader, rnn, output, device='cpu'):
    # check if load existing model
    if args.load:
        fname = os.path.join(args.graph_save_path, f'{args.fname}lstm_epoch_{args.load_epoch}.dat')
        rnn.load_state_dict(torch.load(fname))
        if isinstance(output, dict):
            for key in output:
                fname = os.path.join(args.model_save_path, f'{args.fname}output_{key}_epoch_{args.load_epoch}.dat')
                output[key].load_state_dict(torch.load(fname))
        else:
            fname = os.path.join(args.model_save_path, f'{args.fname}output_epoch_{args.load_epoch}.dat')
            output.load_state_dict(torch.load(fname))
        args.lr = 0.00001
        epoch = args.load_epoch
        print(f'Model loaded! lr: {args.lr}')
    else:
        epoch = 1

    # initialize optimizer
    optim_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    scheduler_rnn = MultiStepLR(optim_rnn, milestones=args.milestones, gamma=args.lr_rate)

    if isinstance(output, dict):
        optim_output = {key: optim.Adam(list(output[key].parameters()), lr=args.lr) for key in output}
        scheduler_output = {key: MultiStepLR(optim_output[key], milestones=args.milestones, gamma=args.lr_rate) for key in optim_output}
    else:
        optim_output = optim.Adam(list(output.parameters()), lr=args.lr)
        scheduler_output = MultiStepLR(optim_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        if args.note.startswith('GraphRNN_VAE'):
            train_vae_epoch(epoch, args, rnn, output, data_loader, optim_rnn, optim_output, scheduler_rnn, scheduler_output, device=device)
        elif args.note == 'GraphRNN_MLP':
            train_mlp_epoch(epoch, args, rnn, output, data_loader, optim_rnn, optim_output, scheduler_rnn, scheduler_output, device=device)
        elif args.note == 'GraphRNN_RNN':
            train_rnn_epoch(epoch, args, rnn, output, data_loader, optim_rnn, optim_output, scheduler_rnn, scheduler_output, device=device)
        time_end = tm.time()
        time_all[epoch-1] = time_end - time_start

        # test (graph generation)
        if (epoch == 1 or epoch % args.epochs_test == 0) and epoch >= args.epochs_test_start:
            for sample_time in range(1, 4):
                G_pred = []
                while len(G_pred) < args.test_total_size:
                    if args.note.startswith('GraphRNN_VAE'):
                        G_pred_step = test_vae_epoch(epoch, args, rnn, output, args.test_batch_size, sample_time=sample_time, device=device)
                    elif args.note == 'GraphRNN_MLP':
                        G_pred_step = test_mlp_epoch(epoch, args, rnn, output, args.test_batch_size, sample_time=sample_time, device=device)
                    elif args.note == 'GraphRNN_RNN':
                        G_pred_step = test_rnn_epoch(epoch, args, rnn, output, args.test_batch_size, device=device)
                    G_pred.extend(G_pred_step)

                # save graphs
                fname = os.path.join(args.graph_save_path, f'{args.fname_pred}epoch_{epoch}_sample_time_{sample_time}.dat')
                save_graph_list(G_pred, fname)
                draw_graph_list(G_pred[:16], 4, 4, fname=f'figures/{args.fname_pred}epoch_{epoch}_sample_time_{sample_time}')

                if args.note == 'GraphRNN_RNN':
                    break
            print('test finished, graphs saved!')

        # save model checkpoint
        if args.save:
            if epoch == 1 or epoch % args.epochs_save == 0:
                fname = os.path.join(args.model_save_path, f'{args.fname}lstm_epoch_{epoch}.dat')
                torch.save(rnn.state_dict(), fname)
                if isinstance(output, dict):
                    for key in output:
                        fname = os.path.join(args.model_save_path, f'{args.fname}output_{key}_epoch_{epoch}.dat')
                        torch.save(output[key].state_dict(), fname)
                else:
                    fname = os.path.join(args.model_save_path, f'{args.fname}output_epoch_{epoch}.dat')
                    torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(os.path.join(args.timing_save_path, args.fname), time_all)


# for graph completion task
def train_graph_completion(args, dataset_test, rnn, output, device='cpu'):
    fname = os.path.join(args.model_save_path, f'{args.fname}lstm_epoch_{args.load_epoch}.dat')
    rnn.load_state_dict(torch.load(fname))
    if isinstance(output, dict):
        for key in output:
            fname = os.path.join(args.model_save_path, f'{args.fname}output_{key}_epoch_{args.load_epoch}.dat')
            output[key].load_state_dict(torch.load(fname))
    else:
        fname = os.path.join(args.model_save_path, f'{args.fname}output_epoch_{args.load_epoch}.dat')
        output.load_state_dict(torch.load(fname))
    epoch = args.load_epoch
    print(f'Model loaded! Epoch: {args.load_epoch}')

    for sample_time in range(1, 4):
        if args.note.startswith('GraphRNN_VAE'):
            G_pred = test_vae_partial_epoch(epoch, args, rnn, output, dataset_test, sample_time=sample_time, device=device)
        elif args.note == 'GraphRNN_MLP':
            G_pred = test_mlp_partial_simple_epoch(epoch, args, rnn, output, dataset_test, sample_time=sample_time, device=device)
        elif args.note == 'GraphRNN_RNN':
            G_pred = test_rnn_partial_simple_epoch(epoch, args, rnn, output, dataset_test, sample_time=sample_time, device=device)

        # save graphs
        fname = os.path.join(args.graph_save_path, f'{args.fname_pred}epoch_{epoch}_sample_time_{sample_time}_graph_completion.dat')
        save_graph_list(G_pred, fname)
        draw_graph_list(G_pred[:16], 4, 4, fname=f'figures/{args.fname_pred}epoch_{epoch}_sample_time_{sample_time}_graph_completion')
    print(f'[{strftime("%Y-%m-%d %H:%M:%S", localtime())}] Graph completion finished. Graphs saved!')


# for NLL evaluation
def train_nll(args, dataset_train, dataset_test, rnn, output, graph_validate_len, graph_test_len, max_iter=1000, device='cpu'):
    fname = os.path.join(args.model_save_path, f'{args.fname}lstm_epoch_{args.load_epoch}.dat')
    rnn.load_state_dict(torch.load(fname))
    if isinstance(output, dict):
        for key in output:
            fname = os.path.join(args.model_save_path, f'{args.fname}output_{key}_epoch_{args.load_epoch}.dat')
            output[key].load_state_dict(torch.load(fname))
    else:
        fname = os.path.join(args.model_save_path, f'{args.fname}output_epoch_{args.load_epoch}.dat')
        output.load_state_dict(torch.load(fname))

    epoch = args.load_epoch
    print(f'Model loaded!, epoch: {args.load_epoch}')
    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write(str(graph_validate_len) + ',' + str(graph_test_len)+'\n')
        f.write('train,test\n')
        for it in range(max_iter):
            if args.note == 'GraphRNN_MLP':
                nll_train = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_train, device=device)
                nll_test = train_mlp_forward_epoch(epoch, args, rnn, output, dataset_test, device=device)
            if args.note == 'GraphRNN_RNN':
                nll_train = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_train, device=device)
                nll_test = train_rnn_forward_epoch(epoch, args, rnn, output, dataset_test, device=device)
            print('train', nll_train, 'test', nll_test)
            f.write(str(nll_train)+',' + str(nll_test)+'\n')
    print('NLL evaluation finished.')
