import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BootHardLoss(nn.Module):
    def __init__(self, beta=0.8):
        super(BootHardLoss, self).__init__()
        self.beta = beta

    def forward(self, logits, targets):
        y_pred = F.softmax(logits, dim=1)
        y_pred = torch.clamp(y_pred, min=torch.finfo(logits.dtype).eps, max=1 - torch.finfo(logits.dtype).eps)
        
        y_true = F.one_hot(targets, num_classes=y_pred.size(1)).float().to(logits.device)
        pred_labels = F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.size(1)).float()

        loss = (self.beta * y_true + (1. - self.beta) * pred_labels) * torch.log(y_pred)
        loss = -loss.sum(dim=1).mean()  # Sum across classes and average over batch

        return loss
