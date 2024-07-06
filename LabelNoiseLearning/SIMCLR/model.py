import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRMLPNet(nn.Module):
    def __init__(self, num_features, num_classes, projection_dim=128, dataset=None):
        super(SimCLRMLPNet, self).__init__()
        # Define the network layers based on the dataset
        self.fc1 = nn.Linear(num_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, num_classes)

        self.projection = nn.Sequential(
            nn.Linear(num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        features = self.fc4(x)
        projections = self.projection(features)
        return features, projections

    def _initialize_weights(self):
        # Initialize weights and biases for all fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
