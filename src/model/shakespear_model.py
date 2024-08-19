import logging

import torch.nn as nn


class RNNShakespeare(nn.Module):
    def __init__(self, num_classes, n_hidden):
        super(RNNShakespeare, self).__init__()
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.embedding = nn.Embedding(self.num_classes, 8)
        self.lstm = nn.LSTM(8, self.n_hidden, batch_first=True, num_layers=2)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        logging.info(f"model instantiation")

    def forward(self, x):
        x = self.embedding(x)
        self.lstm.flatten_parameters()  # Add this line
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

