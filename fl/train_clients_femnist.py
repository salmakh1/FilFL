import os
import random
from collections import Counter

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader

from selection.bandits import rgl, optimized_rgl
from src.data.dataloader_utils import dataloader
from src.data.open_image_utils import select_dataset

log = logging.getLogger(__name__)
from hydra.utils import instantiate
import copy
import time
from src.utils.train_utils import average_weights, test, initialize_weights, set_seed
import wandb
import datetime
import logging
import numpy as np

log = logging.getLogger(__name__)

class trainClientsFemnist(object):

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
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_number

    def train_clients(self):

        ################INITIATE wandb#################

        if self.cfg.use_wandb:
            lr = str(
                self.cfg.client_training.learning_rate)
            run_name = "Clt" + "_" + str(
                self.cfg.num_clients) + "BY" + str(self.cfg.m) + "_" + "Rds" + "_" + str(
                self.cfg.rounds) + "_" + "LclS" + "_" + str(
                self.cfg.client_training.local_steps) + "_Data_" + str(
                self.hydra_cfg["datamodule"]) + "_Splt_" + str(
                self.hydra_cfg["split"]) + "_Bdt_" + str(self.cfg.bandit) + '_' + "Seed_" + str(
                self.cfg.datamodule.seed) + "_LR_" + str(self.cfg.optim.lr) + "_Rand_" + str(
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
        logging.info(f"the GPU device is {torch.cuda.current_device()}")


        # Loading the data
        log.info("################  Instanciating the data  ##############")
        print(OmegaConf.to_yaml(self.cfg.datamodule))
        datamodule = instantiate(self.cfg.datamodule, _recursive_=False)

        # ############## load and split the data for clients #################
        # self.cfg.model.num_classes = datamodule.num_classes

        random.seed(self.cfg.datamodule.seed)
        torch.manual_seed(self.cfg.datamodule.seed)

        # set seed
        set_seed(seed=self.cfg.datamodule.seed)

        self.model = instantiate(self.cfg.model)
        self.model.apply(initialize_weights)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model = self.model.to(device=self.device)
        print(self.model)
        # train and test data:
        log.info("load_and_split")

        clients_ids, train_sets, test_sets, public_data, val_sets, data_weights = datamodule.setup_data("femnist", model=None,
                                                                                          use_val_set=False)


        # Prepare clients and data
        clients = []
        # pub_data
        public_data = dataloader(public_data["x"], public_data["y"])

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
            train_data = dataloader(train_sets[client_id]["x"], train_sets[client_id]["y"])
            test_data = dataloader(test_sets[client_id]["x"], test_sets[client_id]["y"])
            val_data = dataloader(val_sets[client_id]["x"], val_sets[client_id]["y"])

            train_loader = DataLoader(
                train_data,
                batch_size=self.cfg.datamodule.batch_size,
                shuffle=True,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
                collate_fn=None,
            )
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
                cond = t % self.cfg.environment_change == 0  # used for divfl

            if cond:
                available_clients = instantiate(self.cfg.selection, clients=clients,
                                                # clients=range(self.cfg.num_clients),
                                                num_selected_clients=self.cfg.available_clients, t=t)
                if self.cfg.bandit:
                    active_clients = available_clients
                    active_clients_bandit = available_clients
                    active_clients_indices = [client.client_id for client in active_clients]
                    weights_of_active_clients = {client_idx: weight for client_idx, weight in data_weights.items() if
                                                 client_idx in active_clients_indices}
                    logging.info(f"weights_of_active_clients {weights_of_active_clients}")
                    sorted_weights = sorted(weights_of_active_clients.items(), key=lambda x: x[1], reverse=True)
                    client_with_largest_data = sorted_weights[0][0]
                    logging.info(f"client_with_largest_data and data is {sorted_weights[0]}")
                    client_object = [client for client in active_clients if
                                     client.client_id == client_with_largest_data]
                    logging.info(f" the client is {client_object}")
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
                    selected_clients = x

                else:
                    selected_clients = instantiate(self.cfg.selection, clients=x,
                                                   num_selected_clients=self.cfg.m, t=t)

                for client in selected_clients:
                    active_clients.append(clients[client])

            if t % self.cfg.periodicity_of_bandit == 0 and self.cfg.bandit:
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
                    results = client.train(self.model, mu=1)
                else:
                    results = client.train()

                train_acc.append(results["train_acc"])
                train_loss.append(results["train_loss"])

                if not cond or not self.cfg.bandit:
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
                    # logging.info(f"this is round {t}, we are inside the bandit and b is {b}")
                    # logging.info(f"the length of active clients is {len(active_clients)} and it is {active_clients}")
                    if self.cfg.client_filtering:
                        x, bandit_reward, random_reward, ps = optimized_rgl(model=client_object[0].model,
                                                                            local_weights=local_weights,
                                                                            active_clients=active_clients,
                                                                            device=self.device,
                                                                            data_set=client_object[0].train_set,
                                                                            p_bandit=1,
                                                                            initial_reward=bandit_reward,
                                                                            randomized=self.cfg.randomized)
                    else:
                        x, bandit_reward, random_reward, ps = optimized_rgl(model=self.model,
                                                                            local_weights=local_weights,
                                                                            active_clients=active_clients,
                                                                            device=self.device, data_set=pub_data,
                                                                            p_bandit=1,
                                                                            initial_reward=bandit_reward,
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
                        val_results = clients[index].validation()
                        val_acc.append(val_results["val_acc"])
                        val_loss.append(val_results["val_loss"])
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
                global_test_results = test(self.model, self.device, client.test_set)
                tests_acc.append(global_test_results["global_val_acc"])
                tests_loss.append(global_test_results["global_val_loss"])

            # global_test_results = test(self.model, self.device, global_test_set)

            ########### WANDB LOG ###############
            val_acc_batch.append(sum(val_acc) / len(val_acc))
            val_loss_batch.append(sum(val_loss) / len(val_loss))
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
