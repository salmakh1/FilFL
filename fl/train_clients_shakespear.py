import os
import random
from collections import Counter

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import logging
import torch
from torch import nn

from selection.bandits import rgl, optimized_rgl
from src.data.dataloader_utils import dataloader
from src.data.open_image_utils import select_dataset
from src.data.shakespeare_utils import process_x, process_y

log = logging.getLogger(__name__)
from hydra.utils import instantiate
import copy
import time
from src.utils.train_utils import average_weights, test, initialize_weights, set_seed
import wandb
import datetime
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class trainClientsShakespeare(object):

    def __init__(self, num_clients, m, available_clients, cfg, hydra_cfg):

        # self.device = 'cuda' if 'gpu' else 'cpu'
        self.num_clients = num_clients
        self.m = m
        self.available_clients = available_clients
        # self.device = 'cuda' if 'gpu' else 'cpu'
        self.cfg = cfg
        self.device = torch.device('cuda') if cfg.device else torch.device('cpu')
        print(self.device)
        self.training_sets = self.test_dataset = None
        self.model = None
        self.epoch = 0
        self.client_id = 0
        self.hydra_cfg = hydra_cfg

    def train_clients(self):

        ################INITIATE wandb#################
        if self.cfg.use_wandb:
            lr = str(
                self.cfg.client_training.learning_rate)
            # +"_LR_" + str(self.cfg.optim.lr)
            run_name = "Clt" + "_" + str(
                self.cfg.num_clients) + "BY" + str(self.cfg.m) + "_" + "Rds" + "_" + str(
                self.cfg.rounds) + "_" + "LclS" + "_" + str(
                self.cfg.client_training.local_steps) + "_" + "Data" + "_" + str(
                self.hydra_cfg["datamodule"]) + "_Splt_" + str(
                self.hydra_cfg["split"]) + "_Bandit_" + str(self.cfg.bandit) + '_' + "Seed_" + str(
                self.cfg.datamodule.seed) + "_Opt_" + str(self.hydra_cfg["optim"]) + str(
                self.cfg.randomized) + "_power_" + str(self.cfg.power)

            time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H%M%S')
            print(wandb.config)
            if self.cfg.bandit:
                self.cfg.logger.group = "bandit"
                wandb_run = instantiate(self.cfg.logger, id=run_name + "_TS_" + time_stamp)
            else:
                self.cfg.logger.group = "non_bandit"
                wandb_run = instantiate(self.cfg.logger, id=run_name + "_TS_" + time_stamp)
            print(self.cfg.logger)
            self.cfg.client_training.client_id = self.client_id
            wandb.config = OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=True
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.cfg.gpus))
        logging.info(f"{torch.cuda.device_count()}")

        # Loading the data
        logging.info(f"the GPU device is {torch.cuda.current_device()}")
        # # Loading the data
        log.info("################  Instanciating the data  ##############")
        print(OmegaConf.to_yaml(self.cfg.datamodule))
        datamodule = instantiate(self.cfg.datamodule, _recursive_=False)

        # ############## load and split the data for clients #################
        # self.cfg.model.num_classes = datamodule.num_classes

        random.seed(self.cfg.datamodule.seed)
        torch.manual_seed(self.cfg.datamodule.seed)

        # set seed
        set_seed(seed=self.cfg.datamodule.seed)
        logging.info(f"setting up the seed")

        self.model = instantiate(self.cfg.model)

        self.model.apply(initialize_weights)
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model = self.model.to(device=self.device)
        logging.info(f"model instantiation finished")
        # print(self.model)

        # train and test data:
        log.info("load_and_split")
        clients_ids, train_sets, test_sets, public_data, val_sets = datamodule.setup_data("shakespeare",
                                                                                          use_val_set=False)

        # Prepare clients and data
        clients = []
        # pub_data
        logging.info(f" public data is {public_data}")
        # logging.info(f"length of the data is {len(public_data['x'][0])}, {len(public_data['x'][1])}, {len(public_data['x'][2])}")
        public_data = TensorDataset(process_x(public_data["x"]), process_y(public_data["y"]))

        pub_data = DataLoader(
            public_data,
            batch_size=16,  # self.cfg.datamodule.batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            pin_memory=True,
            collate_fn=None,
        )

        for client_id in clients_ids:
            train_data = TensorDataset(process_x(train_sets[client_id]["x"]), process_y(train_sets[client_id]["y"]))
            test_data = TensorDataset(process_x(test_sets[client_id]["x"]), process_y(test_sets[client_id]["y"]))
            val_data = TensorDataset(process_x(val_sets[client_id]["x"]), process_y(val_sets[client_id]["y"]))

            train_loader = DataLoader(
                train_data,
                batch_size=self.cfg.datamodule.batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=True,
                pin_memory=True,
                collate_fn=None,
            )

            if len(train_loader) < 2:
                continue

            test_loader = DataLoader(
                test_data,
                batch_size=16,  # self.cfg.datamodule.batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
                collate_fn=None,
            )
            val_loader = DataLoader(
                val_data,
                batch_size=16,  # self.cfg.datamodule.batch_size,
                shuffle=False,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
                collate_fn=None,
            )
            self.cfg.client_training.client_id = self.client_id

            if self.cfg.fedprox:
                client = instantiate(self.cfg.client_training, device=self.device,
                                     train_loaders=train_loader,
                                     model=self.model,
                                     val_loaders=val_loader, test_set=test_loader,
                                     FedProx=True)
            else:

                client = instantiate(self.cfg.client_training,
                                     device=self.device,
                                     train_loaders=train_loader,
                                     model=self.model,
                                     val_loaders=val_loader,
                                     test_set=test_loader
                                     )
            clients.append(client)
            self.client_id += 1

        self.client_id = 0
        logging.info(f"size of clients list {len(clients)}")
        # TRAINING num_clients  for t rounds

        log.info("#################### Start training  #########################")
        clients_count_filtered = []
        bandit_reward = 0

        logging.info(f"##### The Number of Clients is {len(clients)} #####")

        active_clients_bandit=[]
        for t in range(self.cfg.rounds):

            log.info("####### This is ROUND number {}".format(t))
            b = False
            self.client_id = 0
            train_loss_batch = []
            val_loss_batch = []
            train_acc_batch = []
            val_acc_batch = []
            active_clients = []
            clients_count_filtered = []

            log.info("####### This is round starts with number of clients {}".format(self.cfg.m))
            if not self.cfg.divfl:
                cond = t % self.cfg.environment_change == 0 or t < self.cfg.environment_change
            else:
                cond = t % self.cfg.environment_change == 0

            if cond:
                available_clients = instantiate(self.cfg.selection, clients=clients,
                                                num_selected_clients=self.cfg.available_clients, t=t)
                if self.cfg.bandit:
                    active_clients = available_clients
                    active_clients_bandit = available_clients

                b = True

            if not self.cfg.bandit:
                if self.cfg.power:
                    losses = {}
                    for client in available_clients:
                        val_results = client.validation()
                        losses[client] = val_results["val_loss"]
                    losses = sorted(losses.items(), key=lambda x: x[1], reverse=True)
                    losses = losses[:self.cfg.m]
                    for key in losses:
                        active_clients.append(key[0])
                    logging.info(f"power of choice active clients {len(active_clients)}")

                else:
                    selected_clients = instantiate(self.cfg.selection, clients=available_clients,
                                                   num_selected_clients=self.cfg.m, t=t)
                    active_clients = selected_clients

            if self.cfg.bandit and b == False:
                if len(x) <= self.cfg.m:
                    # selected_clients = [clients[client_id] for client_id in x]
                    logging.info(f" enter the condition x is {x}")
                    selected_clients = x
                else:
                    selected_clients = instantiate(self.cfg.selection, clients=x,
                                                   num_selected_clients=self.cfg.m, t=t)
                    logging.info(f"selected clients = {selected_clients}")

                for client in selected_clients:
                    active_clients.append(clients[client])

            if t % self.cfg.periodicity_of_bandit == 0:
                active_clients = active_clients_bandit

            val_acc = []
            train_acc = []
            val_loss = []
            train_loss = []

            self.client_id = 0
            local_weights = {}

            logging.info(f"The length of the set of active clients is {len(active_clients)}")
            for client in active_clients:
                logging.info(f"######len of the data is {len(client.train_set)}")

                log.info(" training client {} in round {}".format(client, t))
                # now we will get results for training over one batch size
                if self.cfg.fedprox:
                    results = client.train(self.model, mu=self.cfg.mu)
                else:
                    # results = client.train(self.model)
                    results = client.train()
                train_acc.append(results["train_acc"])
                train_loss.append(results["train_loss"])

                # if not cond or not self.cfg.bandit:
                logging.info(f"validation step..")
                val_results = client.validation()
                val_acc.append(val_results["val_acc"])
                val_loss.append(val_results["val_loss"])
                local_weights[client.client_id] = (copy.deepcopy(results["update_weight"]))  # detach

            train_acc_batch.append(sum(train_acc) / len(train_acc))
            train_loss_batch.append(sum(train_loss) / len(train_loss))



            ########## let the filtering decide ##################
            if self.cfg.bandit:
                if b or t % self.cfg.periodicity_of_bandit == 0:
                    x, bandit_reward, random_reward, ps = optimized_rgl(model=self.model,
                                                                        local_weights=local_weights,
                                                                        active_clients=active_clients,
                                                                        device=self.device, data_set=pub_data,
                                                                        p_bandit=1,
                                                                        initial_reward=bandit_reward,
                                                                        task="NLP",
                                                                        randomized=self.cfg.randomized)

                    logging.info(f"decided set {x}")
                    for index in x:
                        clients_count_filtered.append(index)
                    logging.info(f"length of x is {len(clients_count_filtered)}")

                    selected_local_weights = {}
                    if len(x) < self.cfg.m:
                        selected_clients = x
                    else:
                        selected_clients = instantiate(self.cfg.selection, clients=x,
                                                       num_selected_clients=self.cfg.m, t=t)
                    logging.info(f"selected clients: {selected_clients}")
                    for index in selected_clients:
                        selected_local_weights[clients[index].client_id] = copy.deepcopy(
                            local_weights[clients[index].client_id])
                    local_weights = selected_local_weights
                    active_clients = []
                    for indx in selected_clients:
                        active_clients.append(clients[indx])

            ########## aggregated weights after a batch of training##################
            new_weights = average_weights(local_weights)

            ######## update model weights #######
            self.model.load_state_dict(new_weights)
            if not self.cfg.fedprox:
                logging.info(f"Updating clients weights..")
                for client in clients:
                    client.model.load_state_dict(new_weights)
            ######## global_test #######
            tests_acc = []
            tests_loss = []

            for client in clients:
                global_test_results = test(self.model, self.device, client.test_set, task="NLP")
                tests_acc.append(global_test_results["global_val_acc"])
                tests_loss.append(global_test_results["global_val_loss"])

            # global_test_results = test(self.model, self.device, global_test_set)
            val_acc_batch.append(sum(val_acc) / len(val_acc))
            val_loss_batch.append(sum(val_loss) / len(val_loss))
            ########### WANDB LOG ###############
            print("########## wandb log ##################")
            wandb.log({'round/train_loss': sum(train_loss_batch) / len(train_loss_batch),
                       'round/val_loss': sum(val_loss_batch) / len(val_loss_batch),
                       'round/train_accuracy': sum(train_acc_batch) / len(train_acc_batch),
                       'round/val_accuracy': sum(val_acc_batch) / len(val_acc_batch),
                       'round/global_test_acc': sum(tests_acc) / len(tests_acc),
                       'round/global_test_loss': sum(tests_loss) / len(tests_loss)
                       # 'round/global_test_acc': global_test_results["global_val_acc"],
                       # 'round/global_test_loss': global_test_results["global_val_loss"]
                       },
                      step=t)

            if self.cfg.bandit:
                wandb.log({'client/num_selected_clients': len(x),
                           'reward/reward_bandit': bandit_reward,
                           'reward/reward_random': random_reward,
                           'round/bandit_global_test_acc': global_test_results["global_val_acc"],
                           'round/bandit_global_test_loss': global_test_results["global_val_loss"],
                           # 'round/average_p': sum(ps) / len(ps)
                           },
                          step=t)


            else:
                wandb.log({'client/num_selected_clients': self.cfg.m,
                           },
                          step=t)

        logging.info(f" {clients_count_filtered}")
