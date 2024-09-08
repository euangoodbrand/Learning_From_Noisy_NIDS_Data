import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
from torch.utils.data import DataLoader

class MLPNet(nn.Module):
    def __init__(self, num_features=78, num_classes=15, dataset=None, dropout_rate=0.0):
        super(MLPNet, self).__init__()
        self.dropout_rate = dropout_rate
        
        if dataset == 'windows_pe_real':
            self.fc1 = nn.Linear(num_features, 1024)  
            self.fc2 = nn.Linear(1024, 512)          
            self.fc3 = nn.Linear(512, 512)           
            self.fc4 = nn.Linear(512, num_classes) 
        elif dataset == 'BODMAS':
            self.fc1 = nn.Linear(num_features, 2381)  
            self.fc2 = nn.Linear(2381, 1024)         
            self.fc3 = nn.Linear(1024, 1024)          
            self.fc4 = nn.Linear(1024, num_classes)
        else:
            self.fc1 = nn.Linear(num_features, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 128)
            self.fc4 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
