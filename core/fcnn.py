'''
    Class for a Fully Connected Neural Network
'''

import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        out = out.view(x.size(0), -1)
        return out

