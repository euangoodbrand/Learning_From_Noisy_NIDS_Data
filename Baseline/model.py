import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, num_features=78, num_classes=15):
        super(MLPNet, self).__init__()
        # First Fully-Connected Layer
        self.fc1 = nn.Linear(num_features, 256)
        # Second Fully-Connected Layer
        self.fc2 = nn.Linear(256, 128)
        # Third Fully-Connected Layer
        self.fc3 = nn.Linear(128, num_classes)
        # First Dropout layer
        # self.dropout1 = nn.Dropout(0.1)
        # # Second Dropout layer
        # self.dropout2 = nn.Dropout(0.1)

        # Initialize weights using the Xavier scheme for all dense (fully connected) layers
        self._initialize_weights()

    def forward(self, x):
        # Forward pass through the network followed by activations and dropout layers
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = self.fc3(x)
        return x 

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



# class MLPNet(nn.Module):
#     def __init__(self, num_features=78, num_classes=15):
#         super(MLPNet, self).__init__()
#         self.fc1 = nn.Linear(num_features, 256)  
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512, 128)
#         self.fc4 = nn.Linear(128, num_classes)  

#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x))
#         x = F.leaky_relu(self.fc2(x))
#         x = F.leaky_relu(self.fc3(x))
#         x = self.fc4(x)
#         return x



# class MLPNet(nn.Module):
#     def __init__(self, num_features=78, num_classes=15, dropout_rate=0.5):
#         super(MLPNet, self).__init__()
#         self.fc1 = nn.Linear(num_features, 256)
#         self.bn1 = nn.BatchNorm1d(256)  
#         self.fc2 = nn.Linear(256, 128)
#         self.bn2 = nn.BatchNorm1d(128)  
#         self.fc3 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(dropout_rate)  

#     def forward(self, x):
#         x = F.leaky_relu(self.bn1(self.fc1(x)))
#         x = self.dropout(x) 
#         x = F.leaky_relu(self.bn2(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x
