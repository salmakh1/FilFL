import copy

from typing import Optional
import torch
import logging
import torch.nn.functional
from src.utils.train_utils import test, accuracy, subtract_weights

log = logging.getLogger(__name__)


####a
class Client(object):

    def __init__(self, client_id, local_steps, task, learning_rate, batch_size, topk, device, train_loaders, model,
                 val_loaders: Optional = None, test_set: Optional = None, FedProx: Optional = None):
        self.client_id = client_id
        self.device = device
        self.local_steps = local_steps
        self.task = task
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.topk = topk
        self.FedProx = FedProx
        self.model = copy.deepcopy(model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4 )
        self.criterion = torch.nn.CrossEntropyLoss()
        # train
        self.train_set = train_loaders
        # val
        self.val_set = val_loaders
        self.test_set = test_set

    def train(self, model=None, mu=0):
        for param_group in self.optimizer.param_groups:
            logging.info(f"learning rate is {param_group['lr']} for client {self.client_id}")
        self.model.train()
        if model:
            model.eval()
        ac = []
        los = []
        for completed_steps in range(self.local_steps):
            batch_acc = []
            batch_loss = []
            for batch_idx, (images, label) in enumerate(self.train_set):
                if batch_idx == len(self.train_set) - 1 and len(images) < self.batch_size and len(images) <=1: # handle last in complete batch
                    remaining_samples = self.batch_size - len(images)
                    padding = torch.zeros((remaining_samples,) + images.size()[1:], dtype=images.dtype)
                    images = torch.cat([images, padding], dim=0)
                    pad_target = torch.zeros(remaining_samples, dtype=label.dtype)
                    label = torch.cat([label, pad_target], dim=0)

                inputs, labels = images.to(self.device), label.to(self.device)
                if self.task=="NLP":
                    labels = labels.float()
                self.optimizer.zero_grad()
                # forward + backward + optimize

                # ADD FED_PROX:
                if self.FedProx:
                    outputs = self.model(inputs)
                    w1 = []
                    for param in self.model.parameters():
                        w1.append(param)
                    w2 = []
                    for param in model.parameters():
                        w2.append(param)

                    norm = subtract_weights(w1, w2, self.device)
                    loss = self.criterion(outputs, labels) + ((mu / 2) * norm)

                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                run_loss = loss.item()
                if self.topk:
                    accu = accuracy(outputs, labels, topk=(1, 5))
                    acc = accu[1].item()
                else:
                    _, predicted = outputs.max(1)
                    total = labels.size(0)
                    if self.task != "classification":
                        _, lab = labels.max(1)
                        correct = predicted.eq(lab).sum().item()
                    else:
                        correct = predicted.eq(labels).sum().item()
                    acc = 100. * correct / total
                batch_acc.append(acc)
                batch_loss.append(run_loss)
            ac.append(sum(batch_acc) / len(batch_acc))
            los.append(sum(batch_loss) / len(batch_loss))
        acc = sum(ac) / len(ac)
        run_loss = sum(los) / len(los)
        logging.info(f"client {self.client_id} finished with loss {run_loss} and acc {acc}")
        if self.FedProx:
            results = {'clientId': self.client_id, 'update_weight': self.model.state_dict(), 'train_acc': acc,
                       'train_loss': run_loss}
        else:
            results = {'clientId': self.client_id, 'update_weight': self.model.state_dict(), 'train_acc': acc,
                       'train_loss': run_loss}

        return results

    def validation(self):
        test_results = test(self.model, self.device, self.val_set, test_client=True, client_id=self.client_id,
                            topk=self.topk, task=self.task)

        return test_results
