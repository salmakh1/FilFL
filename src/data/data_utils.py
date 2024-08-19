import copy
import itertools
import random
from collections import defaultdict

import torch
from torch.utils.data import random_split, Subset, Dataset
import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import chain
import logging
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


# This function will split the full data to all clients

def new_getattr(self, name):
    """Search recursively for attributes under self.dataset."""
    dataset = self
    if name[:2] == "__":
        raise AttributeError(name)
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
        if hasattr(dataset, name):
            return getattr(dataset, name)
    raise AttributeError(name)


def split_dataset_train_val(train_dataset, val_split, seed, val_dataset=None):

    targets = train_dataset.targets
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split, stratify=targets, random_state=seed
    )
    train_subset = Subset(train_dataset, indices=train_idx)
    val_subset = Subset(val_dataset if val_dataset else train_dataset, indices=val_idx)
    return train_subset, val_subset


def split_subsets_train_val(subsets, val_precent, seed, val_dataset: Dataset = None):
    """
    split clients subsets into train/val sets
    Args:
        val_dataset: give if you have a val dataset that have different transforms than the train dataset
    """
    log.info(f"here")

    train_sets = []
    val_sets = []
    biased=False
    i=0
    for subset in subsets:
        i=i+1
        if biased:
            indices = np.array(subset.indices) if isinstance(subset, Subset) else np.arange(len(subset))
            targets = subset.dataset.targets if isinstance(subset, Subset) else subset.targets
            targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
            df = pd.DataFrame({"target": targets[indices]}, index=indices)
            label_to_indices = {}
            # Map indices to classes (labels, targets)
            for label, group in df.groupby('target'):
                label_to_indices[label] = group.index
            train_indices, val_indices = train_test_split(subset.indices, test_size=val_precent, random_state=seed)
            val_indices= [x for x in val_indices if x not in label_to_indices[list(label_to_indices.keys())[-1]]]

        else:
            train_indices, val_indices = train_test_split(subset.indices, test_size=val_precent, random_state=seed)
        # logging.info(f"val_indices are {val_indices}")
        train_subset = copy.deepcopy(subset)
        train_subset.indices = train_indices

        if val_dataset:
            val_subset = Subset(val_dataset, indices=val_indices)
        else:
            val_subset = copy.deepcopy(subset)
            val_subset.indices = val_indices

        train_subset.__class__.__getattr__ = new_getattr
        val_subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset

        train_sets.append(train_subset)
        val_sets.append(val_subset)
    # ensure that all indices are disjoints and splits are correct
    assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([s.indices for s in train_sets], 2))
    assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([s.indices for s in val_sets], 2))
    assert all(
        (set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations(
            [s.indices for s in itertools.chain(train_sets, val_sets)],
            2
        )
    )
    return train_sets, val_sets


def random_split_data_to_clients(dataset, num_clients, seed, data_points=0):
    """
    Plain random data split amoung clients
    args:
    dataset: pytorch dataset object
    num_clients: int
    seed: int for fixing the splits
    Returns:
    List of Dataset subset object of length=num_clients
    """

    if data_points == 0:
        ds_len = len(dataset)
        split_sizes = [
            ds_len // num_clients if i != num_clients - 1 else ds_len - (ds_len // num_clients * (num_clients - 1))
            for i in range(num_clients)
        ]
        assert ds_len == sum(split_sizes)
        gen = torch.Generator().manual_seed(seed)  # to preserve the same split everytime
        datasets = random_split(dataset=dataset, lengths=split_sizes, generator=gen)
        assert all(
            (set(p0).isdisjoint(set(p1))) for p0, p1 in itertools.combinations([ds.indices for ds in datasets], 2))
    else:
        a = data_points * num_clients
        indices = np.random.permutation(len(dataset))[:a]
        datasets = Subset(dataset, indices)
    return datasets


####### the parameter missing will identify the number of missing classes##########
# we should split the train in this way but keep the test random split.


def generate_zipf_distribution(train_data, num_clients, missing, a=1.1, size=0):
    targets = train_data.dataset.targets
    x = np.arange(1, max(list(targets)) + 2 - missing)
    weights = x ** (-a)
    weights /= weights.sum()
    bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
    # TODO: make the number of datapoints smaller for every client 250 - 500 datapoints
    if size == 0:
        size = int(len(train_data) / num_clients)
        sample = bounded_zipf.rvs(size=size)
    else:
        sample = bounded_zipf.rvs(size=size)
    return sample


def count_elements_from_distr(sample, targets, missing):
    elemet_counts = []
    sample = list(sample)
    for i in range(1, targets + 1 - missing):
        elemet_counts.append(sample.count(i))
    return elemet_counts


def generate_permutation(num_classes, num_clients, missing):
    labels = [*range(0, num_classes, 1)]
    perms = []
    for i in range(num_clients):
        random.seed(i + 10)
        perms.append(random.sample(labels, len(labels) - missing))
    # perm = itertools.permutations(np.arange(0, num_classes))
    print(perms)
    return perms


def split_data_by_idx(dataset):
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)
    label_to_indices = {}
    # Map indices to classes (labels, targets)
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index
    return label_to_indices


def split_nniid(dataset, num_clients, missing, a, data_points):
    logging.info(f"here  {dataset}")
    targets = torch.tensor(dataset.dataset.targets) if isinstance(dataset, Subset) else dataset.targets
    num_classes = len(targets.unique())
    label_to_indices = split_data_by_idx(dataset)
    logging.info("label_to_indices {}".format(label_to_indices))
    perm = generate_permutation(num_classes, num_clients, missing)
    datasets = []
    clients_data = []
    # d={}
    if data_points == 0:
        data_points = len(targets) / num_clients
    for i in range(num_clients):
        samples = generate_zipf_distribution(dataset, num_clients, missing, a, data_points)
        count_samples = count_elements_from_distr(samples, num_classes, missing)
        log.info("elemet_counts {} for client {}".format(count_samples, i))
        client_data = []
        for j in range(len(count_samples)):
            np.random.seed(i + 10 + j)
            sub = np.random.choice(label_to_indices[perm[i][j]], size=count_samples[j], replace=False)
            sub = list(sub)
            logging.info(sub)
            client_data.append(sub)
        client_data_ = list(chain.from_iterable(client_data))
        clients_data.append(client_data_)

    for data in clients_data:
        datasets.append(Subset(dataset, data))
    print("datasets", datasets)
    return datasets


def dirichlet_split(
        dataset, num_clients, seed, min_dataset_size, transformed_dataset=None, alpha=0.5
):
    np.random.seed(seed)
    freq = None
    min_size = 0
    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
    df = pd.DataFrame({"target": targets[indices]}, index=indices)
    label_to_indices = {}
    for label, group in df.groupby('target'):
        label_to_indices[label] = group.index
    labels, classes_count_ = np.unique(df.target, return_counts=True)
    classes_count = defaultdict(int)
    for label, count in zip(labels, classes_count_):
        classes_count[label] = count
    num_classes = len(df.target.unique())

    N = len(df.target)
    net_dataidx_map = {}
    percentage_client_indices_per_class = defaultdict(dict)

    while min_size < min_dataset_size:
        logging.info(f"min size is smaller than min_dataset_size")
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            # freq[k] = proportions
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            print(min_size)

    sum = 0
    data_weights = {}
    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        # print(len(net_dataidx_map[j]))
        sum += len(net_dataidx_map[j])
        data_weights[j] = len(net_dataidx_map[j])

    class_dis = np.zeros((num_clients, num_classes))

    for j in range(num_clients):
        for m in range(num_classes):
            class_dis[j, m] = int((np.array(targets[idx_batch[j]]) == m).sum())
            percentage_client_indices_per_class[j][m] = class_dis[j, m] / classes_count[m]

        logging.info(
            f"the classes for client {j} are {np.nonzero(class_dis[j, :])[0]}"
            f"and the number is {class_dis[j, np.nonzero(class_dis[j, :])[0]]}")

        # print(class_dis.astype(int))

    datasets = []
    weights = {}
    for client_idx in range(num_clients):
        indices = net_dataidx_map[client_idx]
        weights[client_idx] = len(indices)
        if isinstance(dataset, Subset):
            subset = copy.deepcopy(dataset)
            subset.indices = indices
            datasets.append(subset)
        else:
            subset = Subset(dataset=dataset, indices=indices)
            # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
            datasets.append(subset)
    #
    # logging.info(f"the percentages are {percentage_client_indices_per_class}")
    # logging.info(f"the sum is {sum}")
    log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})

    return datasets, data_weights

# def dirichlet_split(
#         dataset, num_clients, seed, min_dataset_size, transformed_dataset=None, alpha=0.5
# ):
#     np.random.seed(seed)
#     indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
#     targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets
#     targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
#     df = pd.DataFrame({"target": targets[indices]}, index=indices)
#     label_to_indices = {}
#     # Map indices to classes (labels, targets)
#     for label, group in df.groupby('target'):
#         label_to_indices[label] = group.index
#     labels, classes_count_ = np.unique(df.target, return_counts=True)
#     classes_count = defaultdict(int)
#     for label, count in zip(labels, classes_count_):
#         classes_count[label] = count
#     logging.info(f"classes_count {classes_count}")
#     client_indices = defaultdict(list)
#     client_indices_per_class = defaultdict(dict)
#     current_min_size = 0
#
#     while current_min_size < min_dataset_size:
#         for cls_idx in df.target.unique():
#             alpha_ = np.ones(num_clients) * alpha
#             # logging.info(f"alpha {alpha} {alpha[0]}")
#             # for i in range(20):
#             #     alpha[i] = 0.2
#             # logging.info(f"alpha:: {alpha} {alpha[0]}")
#
#             # alpha = np.arange(1, num_clients+1)
#             # alpha = np.log(np.arange(2, num_clients + 2))
#             dist_of_cls_idx_across_clients = np.random.dirichlet(alpha_, size=1)
#             dist_of_cls_idx_across_clients = dist_of_cls_idx_across_clients[0]
#             freq = (
#                            np.cumsum(dist_of_cls_idx_across_clients) * classes_count[cls_idx]
#                    ).astype(int)[:-1]
#             assign = np.split(label_to_indices[cls_idx], freq)
#
#             for client_idx, client_cls_indicies in enumerate(assign):
#                 client_indices[client_idx].extend(client_cls_indicies)
#                 client_indices_per_class[client_idx][cls_idx] = client_cls_indicies
#
#             current_min_size = min([len(client_indices[i]) for i in range(num_clients)])
#     assert len(df) == len([idx for _, indices in client_indices.items() for idx in indices])
#     # assert that there is no intersection between clients indices!
#     assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
#                itertools.combinations([indices for _, indices in client_indices.items()], 2))
#     datasets = []
#     data_weights = {}
#     for client_idx in range(num_clients):
#         indices = client_indices[client_idx]
#         data_weights[client_idx] = len(indices)
#
#         if transformed_dataset:
#             if isinstance(transformed_dataset, Subset):
#                 if client_idx % 10 == 0:
#                     subset = copy.deepcopy(transformed_dataset)
#                     subset.indices = indices
#                     datasets.append(subset)
#                 else:
#                     if isinstance(dataset, Subset):
#
#                         subset = copy.deepcopy(dataset)
#                         subset.indices = indices
#                         datasets.append(subset)
#                     else:
#                         subset = Subset(dataset=dataset, indices=indices)
#                         # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
#                         datasets.append(subset)
#             else:
#                 if client_idx % 10 == 0:
#                     subset = Subset(dataset=transformed_dataset, indices=indices)
#                     # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
#                     datasets.append(subset)
#                 else:
#                     if isinstance(dataset, Subset):
#
#                         subset = copy.deepcopy(dataset)
#                         subset.indices = indices
#                         datasets.append(subset)
#                     else:
#                         subset = Subset(dataset=dataset, indices=indices)
#                         # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
#                         datasets.append(subset)
#         else:
#             if isinstance(dataset, Subset):
#                 subset = copy.deepcopy(dataset)
#                 subset.indices = indices
#                 datasets.append(subset)
#             else:
#                 subset = Subset(dataset=dataset, indices=indices)
#                 # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
#                 datasets.append(subset)
#     return datasets, data_weights



def complex_random_client_data_split(
        dataset, num_clients, seed,
        min_dataset_size,
        dist_of_missing_classes="uniform",
        **kwargs
):
    """
    Data split with specific class distribution among clients
    args:
    dataset: pytorch dataset object, should have targets attribute that return list of that contain the labels in a list,
        this is very important since the split will depend on this list.
    num_clients: int
    seed: int for fixing the splits
    dist_of_missing_classes: how to sample the missing classes. Options are uniform, or
        weighted by the inverse of the frequency of classes
    Returns:
    List of Dataset subset object of length=num_clients
    """

    np.random.seed(seed)

    indices = np.array(dataset.indices) if isinstance(dataset, Subset) else np.arange(len(dataset))
    targets = dataset.dataset.targets if isinstance(dataset, Subset) else dataset.targets

    targets = targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets)

    df = pd.DataFrame({"target": targets[indices]}, index=indices)
    # df = pd.DataFrame(data=dataset.targets, columns=['target'])

    label_to_indices = {}
    # Map indices to classes (labels, targets)
    for label, group in df.groupby('target'):
        label_to_indices[label] = set(group.index)

    labels, classes_count = np.unique(df.target, return_counts=True)

    num_classes = len(labels)

    assert num_clients >= num_classes * 3, "To use the complex random split number clients should be equal or higher " \
                                           "than number of classes * 3 "
    # STEP 1: Distribution of number of missing classes per client

    p = 0.5
    i = 0

    # Truncated Geometric
    random_ks = np.floor(np.log(1 - np.random.rand(num_clients) * (1 - (1 - p) ** num_classes)) / np.log(1 - p))
    labels, counts = np.unique(random_ks, return_counts=True)

    # Loop until the sampled geometric dist matches some conditions.
    while not (np.max(random_ks) == num_classes - 1 and np.min(random_ks) == 0 and len(counts) == num_classes):
        p = p - 0.01 if i % 3000 == 0 else p
        i += 1
        random_ks = np.floor(np.log(1 - np.random.rand(num_clients) * (1 - (1 - p) ** num_classes)) / np.log(1 - p))
        labels, counts = np.unique(random_ks, return_counts=True)
        if i >= 1e7:
            raise ValueError("Try Increasing the number of clients")

    # STEP 2: What are the classes to miss.

    if dist_of_missing_classes == "uniform":
        p = None
    elif dist_of_missing_classes == "freq":
        # Weight or freq of each class, needed to compute the inverse freq.
        classes_weights = [class_freq / sum(classes_count) for class_freq in classes_count]
        p = classes_weights
        print("Warning are you sure you want to use the freq as a weight, and not the inverse of the freq?")
    elif dist_of_missing_classes == "inverse_freq":
        # Weight or freq of each class, needed to compute the inverse freq.
        classes_weights = [class_freq / sum(classes_count) for class_freq in classes_count]
        inverse_weights = [1.0 / w for w in classes_weights]  # Invert all weights
        sum_weights = sum(inverse_weights)
        inverse_weights = [w / sum_weights for w in inverse_weights]  # Normalize them back to a prob dist
        p = inverse_weights
    else:
        raise ValueError("Invalid option for dist_of_missing_classes!")

    classes_to_miss = [
        list(np.random.choice(
            np.arange(0, num_classes),
            size=int(random_k),
            replace=False,
            p=p
        )) for random_k in random_ks
    ]

    logging.info(f"{classes_to_miss} {len(classes_to_miss)}")



    flat_classes_to_miss = [cls_idx for clients in classes_to_miss for cls_idx in clients]
    labels, number_of_times_classes_is_missing = np.unique(flat_classes_to_miss, return_counts=True)
    logging.info(f"flat_classes_to_miss {flat_classes_to_miss} and its length is {len(flat_classes_to_miss)}")
    logging.info(f"number_of_times_classes_is_missing {number_of_times_classes_is_missing} and the sum is {sum(number_of_times_classes_is_missing)}")

    client_indices = defaultdict(list)
    client_indices_per_class = defaultdict(dict)
    percentage_client_indices_per_class= defaultdict(dict)
    for _ in range(1000):
        for cls_idx in df.target.unique():
            num_of_clients_missing_cls_idx = number_of_times_classes_is_missing[cls_idx]
            client_who_have_cls_idx = [client_idx for client_idx in range(num_clients) if
                                       cls_idx not in classes_to_miss[client_idx]]
            num_of_clients_for_cls_idx = len(client_who_have_cls_idx)

            assert num_of_clients_missing_cls_idx + len(client_who_have_cls_idx) == num_clients

            minimum_num_of_datapoints_per_class = 5
            while not minimum_num_of_datapoints_per_class * num_of_clients_for_cls_idx <= len(
                    label_to_indices[cls_idx]):
                minimum_num_of_datapoints_per_class -= 1
                if minimum_num_of_datapoints_per_class == 2:
                    break

            assert minimum_num_of_datapoints_per_class * num_of_clients_for_cls_idx <= len(label_to_indices[cls_idx])

            print(
                f"Minimum number of datapoints of class {cls_idx} per client is {minimum_num_of_datapoints_per_class}")

            # Loop until we get a dirichlet dist that meets some conditions
            for i in np.arange(1, 10000, 0.3):
                dirichlet_alpha = np.ones(len(client_who_have_cls_idx)) * i
                dist_of_cls_idx_across_clients = np.random.dirichlet(dirichlet_alpha, size=1)
                dist_of_cls_idx_across_clients = dist_of_cls_idx_across_clients[0]
                freq = [
                    np.floor(client_percent * classes_count[cls_idx]) for client_percent in
                    dist_of_cls_idx_across_clients
                ]
                if len([x for x in freq if x == 0]) == 0 and min(freq) >= minimum_num_of_datapoints_per_class:
                    break  # end loop when the least freq is 1

            # Update the dist such that it gives the correct probability given that the previous client samples was
            # removed
            update_dist = {}
            for i, client_idx in enumerate(client_who_have_cls_idx):
                update_dist[client_idx] = dist_of_cls_idx_across_clients[i] / \
                                          (1 - sum([dist_of_cls_idx_across_clients[j] for j in range(0, i)]))

            for i, client_idx in enumerate(client_who_have_cls_idx):
                current_available_cls_indices = label_to_indices[cls_idx]
                client_portion_size = len(current_available_cls_indices) * update_dist[client_idx]
                client_selected_idx = np.random.choice(
                    a=list(current_available_cls_indices),
                    size=int(np.floor(client_portion_size)),
                    replace=False
                )
                client_indices[client_idx].extend(client_selected_idx)
                client_indices_per_class[client_idx][cls_idx] = list(client_selected_idx)
                label_to_indices[cls_idx].difference_update(client_selected_idx)


            print(label_to_indices[cls_idx])
            if len(label_to_indices[cls_idx]) != 0:
                # choose a client randomly to give it the remaining
                cls_remaining_indices = label_to_indices[cls_idx]
                client_idx = random.choice(client_who_have_cls_idx)
                client_indices[client_idx].extend(cls_remaining_indices)
                client_indices_per_class[client_idx][cls_idx].extend(cls_remaining_indices)
                label_to_indices[cls_idx].difference_update(cls_remaining_indices)
            print(label_to_indices[cls_idx])
            current_cls_freq = [len(cls_indices) for client, indices in client_indices_per_class.items() for
                                cls_idx2, cls_indices in indices.items() if cls_idx == cls_idx2]

            assert len(current_cls_freq) == len(client_who_have_cls_idx)
            assert sum(current_cls_freq) == classes_count[
                cls_idx], f"Class {cls_idx} count = {classes_count[cls_idx]}, while current count = {sum(current_cls_freq)}"
            assert min(current_cls_freq) >= minimum_num_of_datapoints_per_class - 1

        if min([len(indices) for client, indices in client_indices.items()]) >= min_dataset_size:
            break
        # Reset in case we are going to repeat since the condition about minimum size of a dataset was not met!
        label_to_indices = {}
        # Map indices to classes (labels, targets) again, because the old one was emptied
        for label, group in df.groupby('target'):
            label_to_indices[label] = set(group.index)

        client_indices = defaultdict(list)
        client_indices_per_class = d = defaultdict(lambda: defaultdict(dict))

    assert min([len(indices) for client, indices in client_indices.items()]) >= min_dataset_size, \
        "Failed to genertate a split!"
    datasets = []
    # assert that we distributed all the indices!
    assert len(df) == len([idx for _, indices in client_indices.items() for idx in indices])
    # assert that there is no intersection between clients indices!
    assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
               itertools.combinations([indices for _, indices in client_indices.items()], 2))

    for client_idx in range(num_clients):
        indices = client_indices[client_idx]
        if isinstance(dataset, Subset):
            subset = copy.deepcopy(dataset)
            subset.indices = indices
            datasets.append(subset)
        else:
            subset = Subset(dataset=dataset, indices=indices)
            # subset.__class__.__getattr__ = new_getattr  # trick to get the attrs of the original dataset
            datasets.append(subset)
    log.info("Number of data points per client")
    log.info({f"client_{i}": len(ds.indices) for i, ds in enumerate(datasets)})
    return datasets, percentage_client_indices_per_class
