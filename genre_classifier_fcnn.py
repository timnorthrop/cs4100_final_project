import torch
import torch.nn as nn
import torch.nn.functional as F

class GenreClassifierFCNN(nn.Module):
    def __init__(self):
        super(GenreClassifierFCNN, self).__init__()
        self.fc1 = nn.Linear(58, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x