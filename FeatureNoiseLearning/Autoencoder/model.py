import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, encoding_dim, num_classes, dataset=None):
        super(MLPNet, self).__init__()
        if dataset == 'windows_pe_real':
            self.fc1 = nn.Linear(encoding_dim, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, num_classes)
        elif dataset == 'BODMAS':
            self.fc1 = nn.Linear(encoding_dim, 2381)
            self.fc2 = nn.Linear(2381, 1024)
            self.fc3 = nn.Linear(1024, 1024)
            self.fc4 = nn.Linear(1024, num_classes)
        else:
            self.fc1 = nn.Linear(encoding_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 128)
            self.fc4 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dataset=None):
        super(Autoencoder, self).__init__()
        if dataset == 'windows_pe_real':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 1024),
                nn.ReLU(True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, input_dim),
                nn.ReLU(True)
            )
        elif dataset == 'BODMAS':
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 2381),
                nn.ReLU(True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(2381, 4096),
                nn.ReLU(True),
                nn.Linear(4096, input_dim),
                nn.ReLU(True)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(True),
                nn.Linear(512, 256),
                nn.ReLU(True)
            )
            self.decoder = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(True),
                nn.Linear(512, input_dim),
                nn.ReLU(True)
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
