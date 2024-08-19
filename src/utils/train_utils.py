import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import copy
from random import sample
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score, jaccard_score


def average_weights(w):
    """
    Returns the average of the weights.
    """
    weights = list(w.values())
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[key] += weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))

    return w_avg


def subtract_weights(w1, w2, device):
    """
    Returns the average of the weights.
    """
    sum = torch.tensor(0.).to(device)
    for t1, t2 in zip(w1, w2):
        # logging.info(f"######################## T1 {t1}")
        sum += torch.norm(torch.subtract(t1, t2)) ** 2
    return sum


def weighted_average_weights(w):
    """
    Returns the average of the weights.
    """
    weights = list(w.values())
    w_avg = copy.deepcopy(weights[0])
    for key in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[key] += weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(weights))

    return w_avg


#
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def test(model, device, data_set, test_client=False, client_id=False, topk=False, task="classification"):
    model.eval()  # tells net to do evaluating
    criterion = torch.nn.CrossEntropyLoss()
    batch_acc = []
    batch_loss = []
    # logging.info(f"{len(data_set.dataset)}")
    # logging.info(f"dataset {data_set.test_labels}")
    for batch_idx, (images, label) in enumerate(data_set):
        # logging.info(f"{images.unsqueeze(0).size()} target is {label}")
        data, target = images.to(device), label.to(device)
        if task != "classification":
            target = target.float()
        # logging.info(f" the batch labels are {target}")
        with torch.no_grad():
            output = model(data)
            test_loss = criterion(output, target)
            val_running_loss = test_loss.detach().item()
            if topk:
                # acc, k_f1, k_iou, y_pred = top_k(logits=output.detach().cpu(), y=target.cpu(), k=5)
                accu = accuracy(output, target, topk=(1, 5))
                acc = accu[1].item()
            else:
                _, predicted = output.max(1)
                total = target.size(0)
                if task != "classification":
                    _, lab = target.max(1)
                    correct = predicted.eq(lab).sum().item()
                else:
                    correct = predicted.eq(target).sum().item()
                acc = 100. * correct / total
            batch_acc.append(acc)
            batch_loss.append(val_running_loss)

    acc = sum(batch_acc) / len(batch_acc)
    loss = sum(batch_loss) / len(batch_loss)

    if test_client:
        test_results = {'clientId': client_id, 'val_acc': acc, 'val_loss': loss}
    else:
        test_results = {'global_val_acc': acc, 'global_val_loss': loss}

    return test_results


def random_selection(clients, num_selected_clients, t, seed):
    np.random.seed(seed + t)
    # active_clients_idx = list(np.random.choice(num_clients, selected_clients, replace=False))
    active_clients_idx = sample(clients, num_selected_clients)
    return active_clients_idx


def top_k(logits, y, k: int = 1):
    """
    logits : (bs, n_labels)
    y : (bs,)
    """
    labels_dim = 1
    assert 1 <= k <= logits.size(labels_dim)
    k_labels = torch.topk(input=logits, k=k, dim=labels_dim, largest=True, sorted=True)[1]

    logging.info(f"k_labels {k_labels}")
    # True (#0) if `expected label` in k_labels, False (0) if not
    a = ~torch.prod(input=torch.abs(y.unsqueeze(labels_dim) - k_labels), dim=labels_dim).to(torch.bool)

    logging.info(f"a {a}")

    # These two approaches are equivalent
    # if False:
    y_pred = torch.empty_like(y)
    for i in range(y.size(0)):
        if a[i]:
            y_pred[i] = y[i]
        else:
            y_pred[i] = k_labels[i][0]
        # correct = a.to(torch.int8).numpy()
    # else:
    #     a = a.to(torch.int8)
    #     y_pred = a * y + (1 - a) * k_labels[:, 0]
    #     # correct = a.numpy()

    f1 = f1_score(y_pred, y, average='weighted') * 100
    # acc = sum(correct)/len(correct)*100
    acc = accuracy_score(y_pred, y) * 100

    iou = jaccard_score(y, y_pred, average="weighted") * 100

    return acc, f1, iou, y_pred


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res
