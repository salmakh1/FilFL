import itertools
import json
import logging
from collections import defaultdict, OrderedDict
from random import sample

import numpy as np
import matplotlib.pyplot as plt

import os
from sklearn.model_selection import train_test_split


class Shakespeare():

    def __init__(self, seed, num_classes, batch_size, data_dir):
        self.seed = seed
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_dir = os.path.dirname(eval(data_dir))
        self.subdirectory = "leaf/"
        self.data_dir = os.path.join(self.data_dir, self.subdirectory)
        logging.info(f"the data directory is {self.data_dir}")

    # The idea for public datatset is to use only the uniform split for validation

    def read_dir(self, data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        logging.info(f"clients {clients}")
        return clients, groups, data

    def read_data(self, train_data_dir, test_data_dir):
        logging.info(f"function read data")
        '''parses data in given train and test data directories

        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users

        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        '''
        train_clients, train_groups, train_data = self.read_dir(train_data_dir)
        test_clients, test_groups, test_data = self.read_dir(test_data_dir)
        logging.info(f"the number of train_clients is {len(train_clients)}")
        logging.info(f"the number of train_data is {len(train_data)}")

        # logging.info(f"the number of data_groups is {train_groups}")
        pub_data = OrderedDict()
        val_data = OrderedDict()
        train_dataset = OrderedDict()
        # pub_data_x=[]
        # pub_data_y=[]
        pub_data_x = ['Federated learning has emerged as a promising machine learning paradigm that al',
                      'ows collaborative training across distributed clients while keeping their data ',
                      'ocal. However, the success of federated learning heavily relies on overcoming t',
                      'e challenges of training with a large number of clients and non-iid data, which',
                      'often leads to unstable and slow convergence and suboptimal model performance. ',
                      'o address these challenges, many client selection methods have been proposed to',
                      'optimize partial client participation and mitigate the impact of heterogeneous ',
                      'lients. However, these methods only select participants from the pool of availa',
                      'le clients without considering whether the cohort of clients selected at each r',
                      'und contains the most suitable ones.\nIn this context, we introduce a novel appr',
                      'ach called FilFL, which proposes a client filtering procedure to identify the c',
                      'ients that should be considered at each stage of the training process. FilFL di',
                      'cards clients that are likely to have only marginal improvements in the trained',
                      'model compared to other more promising clients. The assessment of client improv',
                      'ment uses a public dataset held by the FL server to gauge the representativenes',
                      ' of different local client data towards global model performance.\nThe main cont',
                      'ibution of our work lies in proposing a yet unexplored approach to optimize cli',
                      'nt participation in federated learning, based on joint representativeness of th',
                      ' overall data. This approach identifies a subset of collaborative clients that ',
                      're filtered based on their suitability as an addition to the other available cl',
                      'ents. The proposed filtering algorithm discards a client when it is not suitabl',
                      ' for the given stage of the training process but keeps it available for later r',
                      'unds.\nTo filter clients, we define a non-monotone combinatorial maximization pr',
                      'blem, and propose a randomized greedy filtering algorithm that adapts the best ',
                      'heoretical guarantees for offline and online submodular maximization. Our appro',
                      'ch not only promises to improve the convergence and performance of federated le',
                      'rning, but it also ensures the privacy and security of the client data. Overall',
                      ' our work presents a novel and promising solution for optimizing client partici',
                      'ation in federated learning and contributes to advancing the state-of-the-art i',
                      ' this important research direction.\nWe introduce client filtering in FL (or Fil',
                      'L), which incorporates client filtering into the most widely studied FL scheme,',
                      'federated averaging (FedAvg). We first present a combinatorial objective for cl',
                      'ent filtering. We then present the randomized greedy algorithm that periodicall',
                      ' optimizes the objective by selecting a filtered subset of clients to be used f']
        pub_data_y = ['l',
                      'l',
                      'h',
                      ' ',
                      'T',
                      ' ',
                      'c',
                      'b',
                      'o',
                      'o',
                      'l',
                      's',
                      ' ',
                      'e',
                      's',
                      'r',
                      'e',
                      'e',
                      'a',
                      'i',
                      'e',
                      'o',
                      'o',
                      't',
                      'a',
                      'a',
                      ',',
                      'p',
                      'n',
                      'F',
                      ' ',
                      'i',
                      'y',
                      'o']


        pub_data["x"] = pub_data_x
        pub_data["y"] = pub_data_y
        logging.info(f"the number of CLINETS is {len(train_clients)}")
        for u in train_clients:

            array_x = np.array(train_data[u]['x'])
            array_y = np.array(train_data[u]['y'])
            X_train, X_pub, y_train, y_pub = train_test_split(array_x, array_y, test_size=0.000005,
                                                              random_state=32)  # todo: CHECK HERE
            X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size=0.2,
                                                              random_state=32)

            dic = {}
            dic['x'] = list(X_train)
            dic['y'] = list(y_train)
            train_dataset[u] = dic
            val_dic = {}
            val_dic['x'] = X_val
            val_dic['y'] = y_val
            val_data[u] = val_dic

        del train_data

        assert train_clients == test_clients
        assert train_groups == test_groups

        return train_clients, train_groups, train_dataset, test_data, pub_data, val_data

    def setup_data(self, dataset, use_val_set=False):
        logging.info(f"function setup data")

        """Instantiates clients based on given train and test data directories.

        Return:
            all_clients: list of Client objects.
        """
        eval_set = 'test' if not use_val_set else 'val'

        train_data_dir = os.path.join(self.data_dir, 'data', dataset, 'data', 'train')
        test_data_dir = os.path.join(self.data_dir, 'data', dataset, 'data', eval_set)
        logging.info(f"This is the train data directory {train_data_dir}")
        logging.info(f"This is the test data directory {test_data_dir}")

        clients_ids, groups, train_data, test_data, pub_data, val_data = self.read_data(train_data_dir, test_data_dir)

        return clients_ids, train_data, test_data, pub_data, val_data

    def plot_data(self, train_set):
        dataiter = iter(train_set)
        images, labels = dataiter.next()
        plt.imshow(np.transpose(images[6].numpy(), (1, 2, 0)))
        plt.savefig("a.pdf")
