import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, num_features=78, num_classes=15, dataset=None, hidden_size=256, dropout_rate=0.5):
        super(MLPNet, self).__init__()
        
        if dataset == 'windows_pe_real':
            self.fc1 = nn.Linear(num_features, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        elif dataset == 'BODMAS':
            self.fc1 = nn.Linear(num_features, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        else:
            self.fc1 = nn.Linear(num_features, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
            self.fc4 = nn.Linear(hidden_size // 2, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = F.relu(self.dropout(self.fc3(x)))
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)