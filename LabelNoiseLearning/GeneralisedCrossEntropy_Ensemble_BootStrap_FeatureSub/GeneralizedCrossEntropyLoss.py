import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# from https://github.com/AlanChou/Truncated-Loss/tree/master

class GeneralizedCrossEntropyLoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropyLoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        logits = torch.clamp(logits, min=-10, max=10)

        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        # Calculate the Generalized Cross Entropy
        loss = (1 - (Yg ** self.q)) / self.q
        loss = torch.mean(loss)

        return loss
