import os
import torch


# program configuration
class Args:
    def __init__(self):
        # if clean tensorboard
        self.clean_tensorboard = True
        # Which CUDA GPU device is used for training
        # self.cuda = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Which GraphRNN model variant is used.
        self.note = 'GraphRNN_MLP'  # the simple version of Graph RNN
        # self.note = 'GraphRNN_RNN'  # the dependent Bernoulli sequence version of GraphRNN

        # for comparison, removing the BFS component
        # self.note = 'GraphRNN_MLP_nobfs'
        # self.note = 'GraphRNN_RNN_nobfs'

        # Which dataset is used to train the model
        # self.graph_type = 'DD'
        # self.graph_type = 'caveman'
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'caveman_small_single'
        # self.graph_type = 'community4'
        # self.graph_type = 'grid'
        self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'
        # self.graph_type = 'enzymes'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer'
        # self.graph_type = 'citeseer_small'

        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type + str(self.noise)

        self.train_size_frac = 0.8  # what fraction of the original data will correspond to training data
        self.valid_size_frac = 0.2  # what fraction of the original data will correspond to validation data

        # if None, then auto calculate
        self.max_num_node = None  # max number of nodes in a graph
        self.max_prev_node = None  # max number of previous nodes that we look back

        # This argument controls whether there are nodes of different types or just one (in that case it is set to None)
        # Set it to something other than None, if there is more than one type of nodes
        self.vocab_size_node_label = None  # 10  # None, 10

        # network config
        # GraphRNN
        self.parameter_shrink = 2 if 'small' in self.graph_type else 1
        self.hidden_size_rnn = int((128 if self.vocab_size_node_label is None else 384) / self.parameter_shrink)  # main RNN hidden size
        self.hidden_size_rnn_output = 16  # hidden size for output RNN
        self.embedding_size_rnn = int((64 if self.vocab_size_node_label is None else 128) / self.parameter_shrink)  # RNN input size
        self.embedding_size_node_label = int(64 / self.parameter_shrink)  # the embedding size for a node's label
        self.embedding_size_rnn_output = 8  # the embedding size for output RNN
        self.embedding_size_output = int(64/self.parameter_shrink)  # the embedding size for output (VAE/MLP)

        self.batch_size = 32  # 32, 8  # normal: 32, and the rest should be changed accordingly (JiaxuanYou set this to 32)
        self.test_batch_size = 32  # how many test examples to generate in a batch
        self.test_total_size = 1000  # how many test examples to generate in total
        self.num_layers = 4  # number of RNN layers

        # training config
        self.num_workers = 4  # num workers to load data, default 4
        self.batch_ratio = 32  # how many batches are in an epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = 100  # 100, 3000 one epoch means self.batch_ratio x batch_size samples (JiaxuanYou set this to 3000)
        self.epochs_test_start = 10  # 10, 100 (JiaxuanYou set this to 100) start testing after `epochs_test_start` epochs
        self.epochs_test = 10  # 10, 100 (JiaxuanYou set this to 100) test every `epochs_test` epochs
        self.epochs_log = 10  # 10, 100 (JiaxuanYou set this to 100) log every `epochs_log` epochs
        self.epochs_save = 10  # save model every `epochs_save` epochs

        self.lr = 0.003
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        self.sample_time = 2  # sample time in each time step, when validating

        # output config
        # self.dir_input = '/dfs/scratch0/
        self.dir_input = './'
        self.model_save_path = os.path.join(self.dir_input, 'model_save')  # only for nll evaluation
        self.graph_save_path = os.path.join(self.dir_input, 'graphs')
        self.figure_save_path = os.path.join(self.dir_input, 'figures')
        self.timing_save_path = os.path.join(self.dir_input, 'timing')
        self.figure_prediction_save_path = os.path.join(self.dir_input, 'figures_prediction')
        self.nll_save_path = os.path.join(self.dir_input, 'nll')

        self.load = False  # if load model, default lr is very low
        self.load_epoch = min(3000, self.epochs)
        self.save = True

        # baseline config
        # self.generator_baseline = 'Gnp'
        self.generator_baseline = 'BA'

        # self.metric_baseline = 'general'
        # self.metric_baseline = 'degree'
        self.metric_baseline = 'clustering'

        # filenames to save intermediate and final outputs
        self.fname = f'{self.note}_{self.graph_type}_{self.num_layers}_{self.hidden_size_rnn}_'
        self.real_graph_filename = None
        self.fname_real = f'{self.fname}real_'
        self.fname_pred = f'{self.fname}pred_'
        self.fname_train = f'{self.fname}train_'
        self.fname_test = f'{self.fname}test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline + '_' + self.metric_baseline
