import os

# program configuration
class Args:
    def __init__(self):
        # if clean tensorboard
        self.clean_tensorboard = True
        # Which CUDA GPU device is used for training
        self.cuda = 1

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

        # if none, then auto calculate
        self.max_num_node = None  # max number of nodes in a graph
        self.max_prev_node = None  # max number of previous nodes that we look back

        # network config
        # GraphRNN
        self.parameter_shrink = 2 if 'small' in self.graph_type else 1
        self.hidden_size_rnn = int(128/self.parameter_shrink)  # hidden size for main RNN
        self.hidden_size_rnn_output = 16  # hidden size for output RNN
        self.embedding_size_rnn = int(64/self.parameter_shrink)  # the size for LSTM input
        self.embedding_size_rnn_output = 8  # the embedding size for output rnn
        self.embedding_size_output = int(64/self.parameter_shrink)  # the embedding size for output (VAE/MLP)

        self.batch_size = 32  # normal: 32, and the rest should be changed accordingly
        self.test_batch_size = 32
        self.test_total_size = 1000
        self.num_layers = 4

        # training config
        self.num_workers = 4  # num workers to load data, default 4
        self.batch_ratio = 32  # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        self.epochs = 3000  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 100
        self.epochs_test = 100
        self.epochs_log = 100
        self.epochs_save = 10

        self.lr = 0.003
        self.milestones = [400, 1000]
        self.lr_rate = 0.3

        self.sample_time = 2  # sample time in each time step, when validating

        # output config
        # self.dir_input = '/dfs/scratch0/jiaxuany0/
        self.dir_input = './'
        self.model_save_path = os.path.join(self.dir_input, 'model_save')  # only for nll evaluation
        self.graph_save_path = os.path.join(self.dir_input, 'graphs')
        self.figure_save_path = os.path.join(self.dir_input, 'figures')
        self.timing_save_path = os.path.join(self.dir_input, 'timing')
        self.figure_prediction_save_path = os.path.join(self.dir_input, 'figures_prediction')
        self.nll_save_path = os.path.join(self.dir_input, 'nll')

        self.load = False  # if load model, default lr is very low
        self.load_epoch = 3000
        self.save = True

        # baseline config
        # self.generator_baseline = 'Gnp'
        self.generator_baseline = 'BA'

        # self.metric_baseline = 'general'
        # self.metric_baseline = 'degree'
        self.metric_baseline = 'clustering'

        # filenames to save intermediate and final outputs
        self.fname = f'{self.note}_{self.graph_type}_{self.num_layers}_{self.hidden_size_rnn}_'
        self.fname_pred = f'{self.fname}pred_'
        self.fname_train = f'{self.fname}train_'
        self.fname_test = f'{self.fname}test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline + '_' + self.metric_baseline
