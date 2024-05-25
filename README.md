---

# Evaluating Dropout for Robustness to Feature Noise

![Feature Noise](https://github.com/euangoodbrand/Learning_From_Noisy_NIDS_Data/raw/main/Assets/image_noise_cleanup2.png)

## Overview
This repository contains the code and datasets used to evaluate the effectiveness of dropout as a regularization technique to improve robustness against feature noise in machine learning models. The project aims to investigate how well dropout can mitigate the impact of feature noise on model performance.

## Introduction

Feature noise is a common issue in real-world datasets, where the input features may be corrupted by various types of noise. This project explores the application of dropout, a popular regularization technique, to see if it can enhance the model's robustness against such noise.

### Dropout

Dropout is a regularization technique that involves randomly dropping units (along with their connections) from the neural network during training. This helps prevent units from co-adapting too much and can improve the generalization of the model.

### Feature Noise

Feature noise involves adding random noise to the input features during training to simulate real-world data corruptions. This project specifically tests the impact of Gaussian noise on the features and evaluates the model's performance with and without dropout.

## Requirements

- Python 3.6+
- PyTorch 1.7.0+
- scikit-learn
- pandas
- numpy
- tqdm

## Implementation

The model architecture and the implementation details are provided below. We use a Multi-Layer Perceptron (MLP) model as the base architecture.

### Model Definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, num_features=78, num_classes=15, dropout_rate=0.5):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after second layer
        x = F.relu(self.fc3(x))
        x = self.dropout(x)  # Apply dropout after third layer
        x = self.fc4(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def add_gaussian_noise(inputs, mean=0.0, std=0.1):
    noise = torch.randn_like(inputs) * std + mean
    return inputs + noise
```

### Training Script

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Example dataset (replace with actual dataset)
train_data = torch.randn(1000, 78)
train_labels = torch.randint(0, 15, (1000,))
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = MLPNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)

def train(model, train_loader, criterion, optimizer, noise_std=0.1):
    model.train()
    for data, target in train_loader:
        data = add_gaussian_noise(data, std=noise_std)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Training the model
for epoch in range(10):
    train(model, train_loader, criterion, optimizer, noise_std=0.1)
```

## Usage

To run the training script and evaluate the model with feature noise and dropout:

```bash
python train.py
```

## Citation

If you find this implementation helpful for your research, please consider citing:

```bash
@article{srivastava2014dropout,
  title={Dropout: A simple way to prevent neural networks from overfitting},
  author={Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan},
  journal={Journal of machine learning research},
  volume={15},
  number={1},
  pages={1929--1958},
  year={2014}
}
```

## Acknowledgments

This project is inspired by various works in the field of machine learning and regularization techniques. Special thanks to the contributors and the community for their continuous support and inspiration.

---
