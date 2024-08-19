import copy
import itertools
import logging

import numpy as np
import random
import torch
from numpy import ndarray

from src.utils.train_utils import test, average_weights


def oracle(model, local_weights, clients, device, data_set, task="classification"):
    """
    Returns the reward of a given list of clients.
    """

    weights = {}

    for client in clients:
        weights[client] = copy.deepcopy(local_weights[client])  # detach

    ########## aggregated weights########
    new_weights = average_weights(weights)

    ######## update model weights #######
    model.load_state_dict(new_weights)

    ######## test with these weights #######
    # logging.info(f" public_data {data_set}")
    if task == "classification":
        test_results = test(model, device, data_set, test_client=False, client_id=False)
    else:
        test_results = test(model, device, data_set, test_client=False, client_id=False, task="NLP")

    ######## oracle #######
    reward = (10 - test_results['global_val_loss'])
    return reward


def rgl(model, local_weights, active_clients, device, data_set):
    """
    Returns the best set of clients.
    """
    active_clients_indexes = []
    for client in active_clients:
        active_clients_indexes.append(client.client_id)

    n = len(active_clients_indexes)

    m = 1  # number of repetitions

    # initial sets
    xi = []
    yi = active_clients_indexes.copy()
    # decide for each arm
    for i in range(0, n):
        ai = 0
        bi = 0

        # estimate the probability
        for j in range(1, m + 1):
            logging.info(f" xi is {xi}")
            if xi == []:
                r1 = 0
            else:
                r1 = oracle(model, local_weights, xi, device, data_set)
            s = set(xi)
            s.add(active_clients_indexes[i])
            r2 = oracle(model, local_weights, list(s), device, data_set)
            ai += (r2 - r1)
            r1 = oracle(model, local_weights, yi, device, data_set)
            if i == 0:
                f_y0 = r1
            s = set(yi)
            s = s - {active_clients_indexes[i]}
            r2 = oracle(model, local_weights, list(s), device, data_set)
            bi += (r2 - r1)

        ai_prime = max(ai / m, 0)
        bi_prime = max(bi / m, 0)

        # decide probabilistically
        if ai_prime == bi_prime:
            if ai_prime == 0:
                p = 1
        else:
            p = ai_prime / (ai_prime + bi_prime)

        logging.info(f"accept with the probability {p}")
        if np.random.uniform(0, 1) < p:
            xi.append(active_clients_indexes[i])
        else:
            yi.remove(active_clients_indexes[i])

    return xi, r1, f_y0


def optimized_rgl(model, local_weights, active_clients, device, data_set, p_bandit=1, initial_reward=0,
                  task="classification", randomized=True):
    """
    With probability p_bandit searches the best set of clients.
    """
    ps = []
    active_clients_indexes = []
    for client in active_clients:
        active_clients_indexes.append(client.client_id)

    n = len(active_clients_indexes)

    m = 1  # number of repetitions

    # initial sets
    xi = []
    yi = active_clients_indexes.copy()

    f_y0 = oracle(model, local_weights, yi, device, data_set, task)

    # with proba p_glob return the whole set
    if random.random() > p_bandit:
        return yi, f_y0, f_y0

    # decide for each arm
    for i in range(0, n):
        ai = 0
        bi = 0

        # estimate the probability
        for j in range(1, m + 1):
            logging.info(f" xi is {xi}")

            if xi == []:
                # r11 = initial_reward
                r11=0
            s = set(xi)
            s.add(active_clients_indexes[i])
            r21 = oracle(model, local_weights, list(s), device, data_set, task)
            ai += (r21 - r11)

            if i == 0:
                r12 = oracle(model, local_weights, yi, device, data_set, task)
                f_y0 = r12

            s = set(yi)
            s = s - {active_clients_indexes[i]}
            r22 = oracle(model, local_weights, list(s), device, data_set, task)
            bi += (r22 - r12)

        add = True
        if randomized:
            ai_prime = max(ai / m, 0)
            bi_prime = max(bi / m, 0)

            # decide probabilistically
            if ai_prime == bi_prime:
                if ai_prime == 0:
                    p = 1
                    # p = 0.5
            else:
                p = ai_prime / (ai_prime + bi_prime)

            ps.append(p)

            logging.info(f"accept with the probability {p}")
            if np.random.uniform(0, 1) < p:
                add = True
                xi.append(active_clients_indexes[i])
                r11 = r21  # x_{i} = x_{i-1} union u_i
                r12 = r12  # y_{i} = y_{i-1}
            else:
                add = False
                yi.remove(active_clients_indexes[i])
                r11 = r11  # x_{i} = x_{i-1}
                r12 = r22  # y_{i} = y_{i-1} minus u_i

        else:
            if ai >= bi:
                logging.info(f"we are in the case ai >= bi where ai= {ai} and bi={bi}")
                add = True
                xi.append(active_clients_indexes[i])
                r11 = r21  # x_{i} = x_{i-1} union u_i
                r12 = r12  # y_{i} = y_{i-1}
            else:
                logging.info(f"we are in the case ai < bi where ai= {ai} and bi={bi}")
                add = False
                r11 = r11  # x_{i} = x_{i-1}
                r12 = r22  # y_{i} = y_{i-1} minus u_i
                yi.remove(active_clients_indexes[i])

        # f(S^{star})
        if add:
            r_star = r21
        else:
            r_star = r22

    return xi, r_star, f_y0, ps

def all_combinations(input_set):
    all_combinations = []
    for i in range(1, len(input_set) + 1):
        combinations = itertools.combinations(input_set, i)
        for c in combinations:
            all_combinations.append(list(c))
    return all_combinations

def optimal_solution(model, local_weights, active_clients, device, data_set, p_bandit=1, initial_reward=0,
                  task="classification", randomized=True):
    """
    With probability p_bandit searches the best set of clients.
    """
    active_clients_indexes = []
    for client in active_clients:
        active_clients_indexes.append(client.client_id)

    # initial sets
    yi = active_clients_indexes.copy()

    OPT = []
    combinations = all_combinations(yi)

    # decide for each arm
    r_max = 0
    for i in range(0, len(combinations)):
        r = oracle(model, local_weights, combinations[i], device, data_set, task)
        if r > r_max:
            r_max = r
            OPT = combinations[i]
    return r_max, OPT


def check_submodularity(active_clients, local_weights, model, device, data):
    gamma = []
    # logging.info(f"{active_clients}")
    active_clients_indexes = []
    for client in active_clients:
        active_clients_indexes.append(client.client_id)
    epochs = 5
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch}")
        for i in range(1, len(active_clients_indexes)):
            random.shuffle(active_clients_indexes)
            A = active_clients_indexes[:i]
            B = active_clients_indexes[i:]

            k = random.randint(1, len(B))
            B = random.sample(B, k=k)
            R1 = 0
            R_A = oracle(model, local_weights, A, device, data)

            for b in B:
                temp_A = A + [b]
                R1 += (oracle(model, local_weights, temp_A, device, data) - R_A)

            R2 = oracle(model, local_weights, A + B, device, data) - R_A
            logging.info(f"#######################################")
            if R1 < 0 and R2 > 0:
                logging.info(f"case 1")
                gamma.append(0)

            if R2 > 0 and R1 > 0:
                logging.info(f"case 2 and {R1 / R2}")
                gamma.append(min(R1 / R2, 1))
                # gamma.append(R1/ R2)

            elif R2 < 0 and R1 < 0:
                logging.info(f"case 3 and {R2 / R1}")
                gamma.append(min(R2 / R1, 1))
                # gamma.append(R2/R1)

            elif R1 > 0 and R2 < 0:
                logging.info(f"case 4")
                gamma.append(1)

            logging.info(f"i is {i} R1 is {R1} and R2 is {R2} and gamma is {gamma[-1]}")
            logging.info(
                f"percentiles 90 {np.percentile(np.array(gamma), 90)}, 75 {np.percentile(np.array(gamma), 75)}, 50 {np.percentile(np.array(gamma), 50)}, 25 {np.percentile(np.array(gamma), 25)}")

            logging.info(f"gamma average is {sum(gamma) / len(gamma)}")

    logging.info(f"last gamma average is {sum(gamma) / len(gamma)}")
    logging.info(
        f"percentiles 90 {np.percentile(np.array(gamma), 90)}, 75 {np.percentile(np.array(gamma), 75)}, 50 {np.percentile(np.array(gamma), 50)}, 25 {np.percentile(np.array(gamma), 25)}")
    logging.info(f"gammas {gamma}")
    percentile_75 = np.percentile(np.array(gamma), 75)
    percentile_50 = np.percentile(np.array(gamma), 50)
    percentile_25 = np.percentile(np.array(gamma), 25)

    return gamma, sum(gamma) / len(gamma), percentile_75, percentile_50, percentile_25


def check_submodularity_percentage(active_clients, local_weights, model, device, data):
    active_clients_indexes = []
    for client in active_clients:
        active_clients_indexes.append(client.client_id)
    epochs = 5

    number_tests = epochs*len(active_clients_indexes)
    gammas = [0.001, 0.2, 0.4, 0.6, 0.8, 1]
    percentage = [0] * len(gammas)

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch}")
        for i in range(1, len(active_clients_indexes)):
            random.shuffle(active_clients_indexes)
            A = active_clients_indexes[:i]
            B = active_clients_indexes[i:]

            k = random.randint(1, len(B))
            B = random.sample(B, k=k)
            R1 = 0
            R_A = oracle(model, local_weights, A, device, data)

            for b in B:
                temp_A = A + [b]
                R1 += (oracle(model, local_weights, temp_A, device, data) - R_A)

            R2 = oracle(model, local_weights, A + B, device, data) - R_A
            logging.info(f"#######################################")

            for j in range(len(gammas)):
                verified = False

                if R1>=gammas[j]*R2 or R1>=R2/gammas[j]:
                    verified = True

                if verified:
                    percentage[j] = percentage[j] + 1/number_tests

            logging.info(f"current percentages {percentage}")
    return percentage


def oracle_divfl(local_weights, available_clients, set_of_clients):
    """
    Returns the reward of a given list of clients.
    """
    reward = 0

    for client in available_clients:
        minimum = 1e20
        distance = 0
        for client_ in set_of_clients:
            for p1, p2 in zip(list(local_weights[client.client_id].values()),
                              list(local_weights[client_].values())):
                distance += torch.norm(p1.type(torch.float) - p2.type(torch.float), p='fro') ** 2
            distance = distance ** (1 / 2)

            if distance < minimum:
                minimum = distance
                # logging.info(f'minimum distance to client {client} is {minimum}')

        reward += minimum

    return reward


def divfl(local_weights, active_clients, k):
    active_clients_indexes = []
    for client in active_clients:
        active_clients_indexes.append(client.client_id)

    # active_clients_indexes = random.sample(active_clients_indexes, 20)

    n = len(active_clients_indexes)

    m = 1
    Xi_ = []

    # decide for each arm
    for _ in range(0, k):
        i_max = 0
        r_max = 0
        for i in range(0, n):
            exist = active_clients_indexes[i] in Xi_
            mean = 0
            if not exist:

                # estimate the probability
                for j in range(1, m + 1):
                    s = set(Xi_)
                    s.add(active_clients_indexes[i])
                    r = oracle_divfl(local_weights, active_clients, list(s))
                    mean += r / m

                if mean >= r_max:
                    r_max = mean
                    i_max = i
        Xi_.append(active_clients_indexes[i_max])

    return Xi_


def optimized_ogl(model, local_weights, active_clients, device, data_set, p_bandit=1):
    """
    With probability p_bandit searches the best set of clients.
    """

    active_clients_indexes = []
    for client in active_clients:
        active_clients_indexes.append(client.client_id)

    n = len(active_clients_indexes)

    m = 1  # number of repetitions

    # initial sets
    xi = []
    yi = active_clients_indexes.copy()

    f_y0 = oracle(model, local_weights, yi, device, data_set)

    # with proba p_glob return the whole set
    if random.random() > p_bandit:
        return yi, f_y0, f_y0

    # decide for each arm
    for i in range(0, n):
        bi = 0

        # estimate the probability
        for j in range(1, m + 1):
            logging.info(f" yi is {yi}")

            if i == 0:
                r12 = f_y0

            s = set(yi)
            s = s - {active_clients_indexes[i]}
            r22 = oracle(model, local_weights, list(s), device, data_set)
            bi += (r22 - r12)

        remove = False
        if bi > 0:
            remove = True
            yi.remove(active_clients_indexes[i])
            r12 = r22

    # f(S^{star})
    if remove:
        r_star = r22
    else:
        r_star = r12

    return yi, r_star, f_y0
