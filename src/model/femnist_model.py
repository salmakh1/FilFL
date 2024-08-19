from torch import nn
import torch.nn.functional as F


# class CNNFemnist(nn.Module):
#     def __init__(self,num_classes):
#         super(CNNFemnist, self).__init__()
#         self.num_classes=num_classes
#         self.conv1 = nn.Conv2d(1, 32, 5, padding="same")
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 5, padding="same")
#         self.fc1 = nn.Linear(64 * 7 *7, 2084)
#         self.fc2 = nn.Linear(2084, 62)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class CNNFemnist(nn.Module):
    def __init__(self,num_classes):
        super(CNNFemnist, self).__init__()
        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(1, 10, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5, padding="same")
        self.fc1 = nn.Linear(20 * 4 *4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 4 *4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)