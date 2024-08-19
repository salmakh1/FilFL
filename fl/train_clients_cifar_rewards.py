import os
import random

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import logging
import torch

from selection.bandits import rgl, optimized_rgl, optimal_solution, oracle

log = logging.getLogger(__name__)
from hydra.utils import instantiate
import copy
import time
from src.utils.train_utils import average_weights, test, random_selection, initialize_weights, set_seed
import wandb
import datetime
import logging

log = logging.getLogger(__name__)


class trainClientsFL(object):

    def __init__(self, num_clients, m, available_clients, cfg, hydra_cfg):

        # self.device = 'cuda' if 'gpu' else 'cpu'
        self.num_clients = num_clients
        self.m = m
        self.available_clients = available_clients
        self.cfg = cfg
        self.hydra_cfg = hydra_cfg
        self.device = torch.device('cuda') if cfg.device else torch.device('cpu')
        print(self.device)
        self.training_sets = self.test_dataset = None
        self.model = None
        self.epoch = 0
        self.client_id = 0
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.gpu_number

    def train_clients(self):

        ################INITIATE wandb#################
        if self.cfg.use_wandb:
            # "_" + "Opt" + "_" + str(self.hydra_cfg["optim"]) + + "_LR_" + lr + "_pubD" + pub
            lr = str(
                self.cfg.client_training.learning_rate)
            # pub = str(self.cfg.datamodule.pub_size)
            run_name = "Clt" + "_" + str(
                self.cfg.num_clients) + "BY" + str(self.cfg.m) + "_" + "Rds" + "_" + str(
                self.cfg.rounds) + "_" + "LclS" + "_" + str(
                self.cfg.client_training.local_steps) + "_" + "Data" + "_" + str(
                self.hydra_cfg["datamodule"]) + "_Bandit_" + str(self.cfg.bandit) + '_' + "Seed_" + str(
                self.cfg.datamodule.seed) + "_Opt_" + str(self.hydra_cfg["optim"]) + "_pub_" + str(
                self.cfg.datamodule.pub_size) + "_LR_" + str(self.cfg.optim.lr) + "_rand_" + str(
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

        logging.info(f"{torch.cuda.device_count()}")

        # Loading the data
        logging.info(f"the GPU device is {torch.cuda.current_device()}")
        log.info("################  Instanciating the data  ##############")
        print(OmegaConf.to_yaml(self.cfg.datamodule))
        datamodule = instantiate(self.cfg.datamodule, _recursive_=False)

        # ############## load and split the data for clients #################
        self.cfg.model.num_classes = datamodule.num_classes

        random.seed(self.cfg.datamodule.seed)
        torch.manual_seed(self.cfg.datamodule.seed)

        # set seed
        set_seed(seed=self.cfg.datamodule.seed)

        self.model = instantiate(self.cfg.model)
        self.model.apply(initialize_weights)

        self.model = self.model.to(device=self.device)
        print(self.model)
        # train and test data:
        log.info("load_and_split")
        train_sets, val_sets, public_data = datamodule.load_and_split(self.cfg)
        pub_data = datamodule.public_loader(public_data)
        global_test_set = datamodule.test_set()

        logging.info(f"WE SUCCESSFULLY LOADED TRAIN, VAL AND TEST {train_sets}")
        # Create clients
        log.info("Preparing the clients and the models and datasets..")
        clients = []
        for i in range(self.cfg.num_clients):
            train_loader = datamodule.train_loaders(train_sets)
            val_loader = datamodule.val_loaders(val_sets)

            # client object
            self.cfg.client_training.client_id = self.client_id

            if self.cfg.fedprox:
                client = instantiate(self.cfg.client_training, device=self.device,
                                     train_loaders=train_loader,
                                     model=self.model,
                                     val_loaders=val_loader, FedProx=True)
            else:

                client = instantiate(self.cfg.client_training, device=self.device,
                                     train_loaders=train_loader,
                                     model=self.model,
                                     val_loaders=val_loader)

            clients.append(client)
            self.client_id += 1
            datamodule.next_client()

        self.client_id = 0
        datamodule.set_client()

        ##TRAINING num_clients  for t rounds

        log.info("#################### Start training  #########################")
        bandit_reward = 0

        ###decay
        schedulers = []
        for client in clients:
            schedulers.append(torch.optim.lr_scheduler.StepLR(client.optimizer, step_size=10, gamma=0.998))

        cumulative_reward_fedAvg = 0
        cumulative_reward_FilFL = 0
        cumulative_reward_random = 0
        unique_selected_clients = set()
        for t in range(self.cfg.rounds):
            log.info("####### This is ROUND number {}".format(t))
            train_clients_times = []
            datamodule.set_client()
            self.client_id = 0
            b = False
            train_loss_batch = []
            train_acc_batch = []
            test_loss_batch = []
            test_acc_batch = []
            active_clients = []
            clients_count_filtered = []
            local_weights = {}
            # select new clients and distribute data among clients
            log.info("####### This is round starts with number of clients {}".format(self.cfg.m))
            if not self.cfg.divfl:
                cond = t % self.cfg.periodicity_of_bandit == 0 or t < self.cfg.periodicity_of_bandit
            else:
                cond = t % self.cfg.periodicity_of_bandit == 0

            if cond:  # used for divfl
                available_clients = instantiate(self.cfg.selection, clients=clients,
                                                # clients=range(self.cfg.num_clients),
                                                num_selected_clients=self.cfg.available_clients, t=t)
                if self.cfg.bandit:
                    active_clients = available_clients
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
                    unique_selected_clients.update(selected_clients)

            if self.cfg.bandit and b == False:
                if len(x) <= self.cfg.m:
                    # selected_clients = [clients[client_id] for client_id in x]
                    selected_clients = x
                else:
                    selected_clients = instantiate(self.cfg.selection, clients=x,
                                                   num_selected_clients=self.cfg.m, t=t)

                unique_selected_clients.update(selected_clients)
                for client in selected_clients:
                    active_clients.append(clients[client])

            logging.info(f"active_clients {len(active_clients)} ")

            val_acc = []
            train_acc = []
            val_loss = []
            train_loss = []

            self.client_id = 0
            local_weights = {}
            # initial_weights = self.model.state_dict()
            logging.info(f"active clients {active_clients}")
            for client in active_clients:
                # self.model.load_state_dict(initial_weights)
                log.info(" training client {} in round {}".format(client, t))
                # now we will get results for training over one batch size
                train_time_start = time.time()
                if self.cfg.fedprox:
                    results = client.train(self.model, mu=1)
                else:
                    results = client.train()
                train_time_end = time.time()
                train_clients_times.append(train_time_end - train_time_start)
                train_acc.append(results["train_acc"])
                train_loss.append(results["train_loss"])
                val_results = client.validation()
                val_acc.append(val_results["val_acc"])
                val_loss.append(val_results["val_loss"])

                local_weights[client.client_id] = (copy.deepcopy(results["update_weight"]))  # detach

            train_acc_batch.append(sum(train_acc) / len(train_acc))
            test_acc_batch.append(sum(val_acc) / len(val_acc))
            train_loss_batch.append(sum(train_loss) / len(train_loss))
            test_loss_batch.append(sum(val_loss) / len(val_loss))

            ########## let the filtering decide ##################
            if self.cfg.bandit:
                if b:
                    # x, bandit_reward, random_reward = rgl(self.model, local_weights, active_clients, self.device, pub_data)
                    logging.info("START COUNTING FOR RGF")
                    time_start = time.time()

                    max_reward, OPT = optimal_solution(model=self.model,
                                                    local_weights=local_weights,
                                                    active_clients=active_clients,
                                                    device=self.device, data_set=pub_data,
                                                    p_bandit=1,
                                                    initial_reward=bandit_reward,
                                                    randomized=self.cfg.randomized)


                    logging.info(f"OPT set {OPT} with reward {max_reward}")

                    rgfx, rgfbandit_reward, random_reward, ps = optimized_rgl(model=self.model,
                                                                        local_weights=local_weights,
                                                                        active_clients=active_clients,
                                                                        device=self.device, data_set=pub_data,
                                                                        p_bandit=1,
                                                                        initial_reward=bandit_reward,
                                                                        randomized=True)

                    logging.info(f"RGF set {rgfx} with reward {rgfbandit_reward}")

                    x, dgfbandit_reward, random_reward, ps = optimized_rgl(model=self.model,
                                                                        local_weights=local_weights,
                                                                        active_clients=active_clients,
                                                                        device=self.device, data_set=pub_data,
                                                                        p_bandit=1,
                                                                        initial_reward=bandit_reward,
                                                                        randomized=False)

                    logging.info(f"DGF set {x} with reward {dgfbandit_reward}")


                    time_end = time.time()
                    time_RGF = time_end - time_start

                    cumulative_reward_FilFL += bandit_reward
                    cumulative_reward_random += random_reward
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

                    unique_selected_clients.update(selected_clients)

                    logging.info(f"selected clients: {selected_clients}")
                    for index in selected_clients:
                        selected_local_weights[clients[index].client_id] = copy.deepcopy(
                            local_weights[clients[index].client_id])
                    local_weights = selected_local_weights

            ########## aggregated weights after a batch of training##################
            time_average_start = time.time()
            new_weights = average_weights(local_weights)
            time_average_end = time.time()
            ######## update model weights #######
            self.model.load_state_dict(new_weights)
            for client in clients:
                client.model.load_state_dict(new_weights)
            ######## global_test #######
            global_test_results = test(self.model, self.device, global_test_set)

            ##########update scheduler ##############
            if t != 0:
                for scheduler in schedulers:
                    scheduler.step()
            ########### WANDB LOG ###############
            print("########## wandb log ##################")
            wandb.log({'round/train_loss': sum(train_loss_batch) / len(train_loss_batch),
                       'round/val_loss': sum(test_loss_batch) / len(test_loss_batch),
                       'round/train_accuracy': sum(train_acc_batch) / len(train_acc_batch),
                       'round/val_accuracy': sum(test_acc_batch) / len(test_acc_batch),
                       'round/global_test_acc': global_test_results["global_val_acc"],
                       'round/global_test_loss': global_test_results["global_val_loss"],
                       'time/client_train_time': max(train_clients_times),
                       'time/time_aggregation': time_average_end - time_average_start
                       },
                      step=t)

            if self.cfg.bandit:
                wandb.log({'client/num_selected_clients': len(x),
                           'reward/reward_dgfbandit': dgfbandit_reward,
                           'reward/reward_rgfbandit': rgfbandit_reward,
                           'reward/reward_random': random_reward,
                           'reward/reward_max': max_reward,
                           'round/bandit_global_test_acc': global_test_results["global_val_acc"],
                           'round/bandit_global_test_loss': global_test_results["global_val_loss"],
                           'round/bandit_train_loss': sum(train_loss_batch) / len(train_loss_batch),
                           # 'round/average_p': sum(ps) / len(ps),
                           # 'reward/cumulative_reward': cumulative_reward_FilFL,
                           # 'reward/cumulative_random_reward': cumulative_reward_random,
                           'client/unique_clients': len(unique_selected_clients),
                           'time/RGF_time': time_RGF,
                           },
                          step=t)

            else:
                wandb.log({'client/num_selected_clients': self.cfg.m,
                           'client/unique_clients': len(unique_selected_clients),
                           'round/non_bandit_global_test_acc': global_test_results["global_val_acc"],
                           'round/non_bandit_global_test_loss': global_test_results["global_val_loss"],
                           'round/non_bandit_train_loss': sum(train_loss_batch) / len(train_loss_batch),

                           # 'reward/cumulative_reward': cumulative_reward_fedAvg,
                           },
                          step=t)
