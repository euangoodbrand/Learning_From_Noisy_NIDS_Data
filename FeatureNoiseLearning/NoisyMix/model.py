import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, num_features=78, num_classes=15, dataset=None):
        super(MLPNet, self).__init__()
        # First Fully-Connected Layer
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

        self._initialize_weights()

    def forward(self, x, targets=None, jsd=0, mixup_alpha=0.0, manifold_mixup=0.0, add_noise_level=0.0, mult_noise_level=0.0, sparse_level=0.0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        outputs = self.fc4(x)

        if mixup_alpha > 0:
            if jsd == 0:
                # Here you can return additional values for mixup
                return outputs, targets, targets, 1.0  # Simplified return for non-JSD mixup
            else:
                # Here you can return additional values for mixup
                return outputs, targets, targets, 1.0  # Simplified return for JSD mixup
        return outputs

    def _initialize_weights(self):
        # Initialize weights and biases for all fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Ensure to initialize weights using the Xavier scheme
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
