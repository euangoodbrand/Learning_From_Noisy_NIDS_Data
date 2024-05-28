from __future__ import print_function 
from model import MLPNet

import os
import re
import csv
import datetime
import argparse
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, ADASYN

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

from loss import loss_coteaching, loss_coteaching_plus

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--label_noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--label_noise_type', type=str, help='Type of noise to introduce', choices=['uniform', 'class', 'feature','MIMICRY'], default='uniform')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type=str, help='cicids', choices=['CIC_IDS_2017','windows_pe_real','BODMAS'])
parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus]', default='coteaching_plus')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--data_augmentation', type=str, choices=['none', 'smote', 'undersampling', 'oversampling', 'adasyn'], default='none', help='Data augmentation technique, if any')
parser.add_argument('--imbalance_ratio', type=float, default=0.0, help='Ratio to imbalance the dataset')
parser.add_argument('--weight_resampling', type=str, choices=['none','Naive', 'Focal', 'Class-Balance'], default='none', help='Select the weight resampling method if needed')
parser.add_argument('--feature_add_noise_level', type=float, default=0.0, help='Level of additive noise for features')
parser.add_argument('--feature_mult_noise_level', type=float, default=0.0, help='Level of multiplicative noise for features')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == "CIC_IDS_2017":
    batch_size = 256
    learning_rate = args.lr 
    init_epoch = 0
elif args.dataset == "windows_pe_real":
    batch_size = 128
    learning_rate = args.lr 
    init_epoch = 0
elif args.dataset == "BODMAS":
    batch_size = 128
    learning_rate = args.lr 
    init_epoch = 0

class CICIDSDataset(Dataset):
    def __init__(self, features, labels, noise_or_not):
        self.features = features
        self.labels = labels
        self.noise_or_not = noise_or_not 

    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return feature, label, index  

    def __len__(self):
        return len(self.labels)

if args.forget_rate is None:
    forget_rate=args.label_noise_rate
else:
    forget_rate=args.forget_rate

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def compute_reconstruction_error(autoencoder, data_loader):
    autoencoder.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].cuda()  # Only take the first element which is the data
            recon_data = autoencoder(data)
            error = ((data - recon_data) ** 2).sum(dim=-1)  # Fixed the dimension
            reconstruction_errors.extend(error.cpu().numpy())

    return np.array(reconstruction_errors)

def select_clean_samples_with_autoencoder(batch, autoencoder, threshold):
    data = batch[0].cuda()  # Only take the first element which is the data
    indices = batch[2]
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, dtype=torch.long).cuda()
    reconstruction_error = compute_reconstruction_error(autoencoder, [(data,)])
    clean_indices = reconstruction_error < threshold
    clean_sample_indices = indices[clean_indices].cpu().numpy()
    return clean_sample_indices


def introduce_noise(labels, features, label_noise_type, label_noise_rate):
    if label_noise_type == 'uniform':
        return introduce_uniform_noise(labels, label_noise_rate)
    elif label_noise_type == 'class':
        return introduce_class_dependent_label_noise(labels, predefined_matrix, label_noise_rate)
    elif label_noise_type == 'feature':
        thresholds = calculate_feature_thresholds(features)
        return introduce_feature_dependent_label_noise(features, labels, label_noise_rate, n_neighbors=5)
    elif label_noise_type == 'MIMICRY':
        return introduce_mimicry_noise(labels, predefined_matrix, label_noise_rate)
    else:
        raise ValueError("Invalid noise type specified.")

def apply_data_augmentation(features, labels, augmentation_method):
    try:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution before augmentation: {dict(zip(unique, counts))}")

        if augmentation_method == 'smote':
            min_samples = np.min(counts)
            n_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
            features, labels = smote.fit_resample(features, labels)
        elif augmentation_method == 'undersampling':
            rus = RandomUnderSampler(random_state=42)
            features, labels = rus.fit_resample(features, labels)
        elif augmentation_method == 'oversampling':
            ros = RandomOverSampler(random_state=42)
            features, labels = ros.fit_resample(features, labels)
        elif augmentation_method == 'adasyn':
            min_samples = np.min(counts)
            n_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            adasyn = ADASYN(random_state=42, n_neighbors=n_neighbors)
            features, labels = adasyn.fit_resample(features, labels)

        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution after augmentation: {dict(zip(unique, counts))}")

        return features, labels
    except ValueError as e:
        print(f"Error during {augmentation_method}: {e}")
        return features, labels
    except NotFittedError as e:
        print(f"Model fitting error with {augmentation_method}: {e}")
        return features, labels

predefined_matrix = np.array([
    [0.8, 0.03, 0.01, 0.01, 0.06, 0.0, 0.0, 0.0, 0.07, 0.0, 0.02, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.01, 0.0, 0.98, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.03, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.0, 0.02, 0.01, 0.0, 0.0],
    [0.0, 0.0, 0.01, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.03, 0.0, 0.0, 0.03, 0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0],
    [0.3, 0.0, 0.0, 0.03, 0.26, 0.0, 0.0, 0.0, 0.05, 0.0, 0.01, 0.35]
])

noise_transition_matrix = np.array([
    [0.534045, 0.005340, 0.001335, 0.012016, 0.093458, 0.006676, 0.009346, 0.016021, 0.088117, 0.072096, 0.125501, 0.036048],
    [0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.501524, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.498476, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.995006, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.004994],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.006250, 0.000000, 0.006250, 0.000000, 0.000000, 0.000000, 0.987500, 0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000],
    [0.000000, 0.000000, 0.000000, 0.019231, 0.730769, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.250000]
])

def feature_noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    device = x.device
    add_noise = torch.zeros_like(x, device=device)
    mult_noise = torch.ones_like(x, device=device)
    scale_factor_additive = 75
    scale_factor_multi = 200

    if add_noise_level > 0.0:
        # Generate additive noise with an aggressive Beta distribution
        beta_add = np.random.beta(0.1, 0.1, size=x.shape)  # Aggressive Beta distribution
        beta_add = torch.from_numpy(beta_add).float().to(device)
        # Scale to [-1, 1] and then apply additive noise
        beta_add = scale_factor_additive * (beta_add - 0.5)  # Scale to range [-1, 1]
        add_noise = add_noise_level * beta_add

    if mult_noise_level > 0.0:
        # Generate multiplicative noise with an aggressive Beta distribution
        beta_mult = np.random.beta(0.1, 0.1, size=x.shape)  # Aggressive Beta distribution
        beta_mult = torch.from_numpy(beta_mult).float().to(device)
        # Scale to [-1, 1] and then apply multiplicative noise
        beta_mult = scale_factor_multi * (beta_mult - 0.5)  # Scale to range [-1, 1]
        mult_noise = 1 + mult_noise_level * beta_mult
    
    return mult_noise * x + add_noise

def introduce_class_dependent_label_noise(labels, class_noise_matrix, label_noise_rate):
    if label_noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    n_samples = len(labels)
    n_noisy = int(n_samples * label_noise_rate)
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    new_labels = labels.copy()
    noise_or_not = np.zeros(n_samples, dtype=bool)

    for idx in noisy_indices:
        original_class = labels[idx]
        new_labels[idx] = np.random.choice(np.arange(len(class_noise_matrix[original_class])), p=class_noise_matrix[original_class])
        noise_or_not[idx] = new_labels[idx] != labels[idx]

    return new_labels, noise_or_not

def introduce_mimicry_noise(labels, class_noise_matrix, label_noise_rate):
    if label_noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    n_samples = len(labels)
    n_noisy = int(n_samples * label_noise_rate)
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    new_labels = labels.copy()
    noise_or_not = np.zeros(n_samples, dtype=bool)

    for idx in noisy_indices:
        original_class = labels[idx]
        new_labels[idx] = np.random.choice(np.arange(len(class_noise_matrix[original_class])), p=class_noise_matrix[original_class])
        noise_or_not[idx] = new_labels[idx] != labels[idx]

    return new_labels, noise_or_not

def calculate_feature_thresholds(features):
    thresholds = np.percentile(features, 50, axis=0)
    return thresholds

def introduce_feature_dependent_label_noise(features, labels, label_noise_rate, n_neighbors=5):
    if label_noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    n_samples = len(labels)
    n_noisy = int(n_samples * label_noise_rate)
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    new_labels = labels.copy()
    noise_or_not = np.zeros(n_samples, dtype=bool)

    for i in noisy_indices:
        neighbor_indices = indices[i][1:]
        neighbor_classes = labels[neighbor_indices]
        different_class_neighbors = neighbor_indices[neighbor_classes != labels[i]]

        if len(different_class_neighbors) > 0:
            chosen_neighbor = np.random.choice(different_class_neighbors)
            new_label = labels[chosen_neighbor]
            new_labels[i] = new_label
            noise_or_not[i] = new_label != labels[i]

    return new_labels, noise_or_not

def introduce_uniform_noise(labels, label_noise_rate):
    if label_noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    n_samples = len(labels)
    n_noisy = int(label_noise_rate * n_samples)
    noisy_indices = np.random.choice(np.arange(n_samples), size=n_noisy, replace=False)

    new_labels = labels.copy()
    noise_or_not = np.zeros(n_samples, dtype=bool)
    unique_labels = np.unique(labels)

    for idx in noisy_indices:
        original_label = labels[idx]
        possible_labels = np.delete(unique_labels, np.where(unique_labels == original_label))
        new_label = np.random.choice(possible_labels)
        new_labels[idx] = new_label
        noise_or_not[idx] = True

    return new_labels, noise_or_not

def apply_imbalance(features, labels, ratio, min_samples_per_class=3, downsample_half=True):
    if ratio == 0:
        print("No imbalance applied as ratio is 0.")
        return features, labels

    if ratio >= 1:
        raise ValueError("Imbalance ratio must be less than 1.")

    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    
    if downsample_half:
        downsample_classes = unique[n_classes // 2:]
        keep_classes = unique[:n_classes // 2]
    else:
        downsample_classes = unique
        keep_classes = []

    keep_indices = []
    keep_class_counts = [count for cls, count in zip(unique, counts) if cls in keep_classes]
    if keep_class_counts:
        average_keep_class_count = int(np.mean(keep_class_counts))
    else:
        average_keep_class_count = min(counts)
    
    indices_to_keep = []
    for cls in unique:
        class_indices = np.where(labels == cls)[0]
        if cls in downsample_classes:
            n_majority_new = max(int(average_keep_class_count * ratio), min_samples_per_class)
            if len(class_indices) > n_majority_new:
                keep_indices = np.random.choice(class_indices, n_majority_new, replace=False)
            else:
                keep_indices = class_indices
        else:
            keep_indices = class_indices

        indices_to_keep.extend(keep_indices)

    indices_to_keep = np.array(indices_to_keep)
    np.random.shuffle(indices_to_keep)
    
    return features[indices_to_keep], labels[indices_to_keep]

def compute_weights(labels, no_of_classes, beta=0.9999, gamma=2.0, device='cuda'):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    samples_per_class = np.bincount(labels, minlength=no_of_classes)

    if args.weight_resampling == 'Naive':
        weights = 1.0 / (samples_per_class + 1e-9)
    elif args.weight_resampling == 'Class-Balance':
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / (np.array(effective_num) + 1e-9)
    elif args.weight_resampling == 'Focal':
        initial_weights = 1.0 / (samples_per_class + 1e-9)
        focal_weights = initial_weights ** gamma
        weights = focal_weights
    elif args.weight_resampling == 'none':
        weights = np.ones(no_of_classes, dtype=np.float32)
    else:
        print(f"Unsupported weight computation method: {args.weight_resampling}")
        raise ValueError("Unsupported weight computation method")

    total_weight = np.sum(weights)
    if total_weight == 0:
        weights = np.ones(no_of_classes, dtype=np.float32)
    else:
        weights = (weights / total_weight) * no_of_classes

    weight_per_label = torch.from_numpy(weights).float().to(device)
    weight_per_label = weight_per_label[torch.from_numpy(labels).to(device)]
    
    return weight_per_label

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)

# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    if fr_type == 'type_1':
        rate_schedule = np.ones(args.n_epoch) * forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    return rate_schedule

rate_schedule = gen_forget_rate(args.fr_type)

save_dir = args.result_dir + '/' + args.dataset + '/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

label_noise_str = f"{args.label_noise_type}-label-noise{args.label_noise_rate}_" if args.label_noise_rate > 0 else ""
model_str = f"{args.model_type}_{args.dataset}_{'no_augmentation' if args.data_augmentation == 'none' else args.data_augmentation}_{label_noise_str}add-noise{args.feature_add_noise_level}_mult-noise{args.feature_mult_noise_level}_imbalance{args.imbalance_ratio}"

txtfile = save_dir + "/" + model_str + ".csv"
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model1, optimizer1, model2, optimizer2, epoch, args, no_of_classes, noise_or_not, autoencoder, threshold):
    model1.train()
    model2.train()
    train_total = 0
    train_correct1 = 0
    train_correct2 = 0
    total_loss1 = 0
    total_loss2 = 0

    epoch_losses1 = []
    epoch_losses2 = []

    for i, (data, labels, indices) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        indices = indices.cpu().numpy().transpose()

        logits1 = model1(data)
        logits2 = model2(data)

        clean_sample_indices_A = select_clean_samples_with_autoencoder((data, labels, indices), autoencoder, threshold)
        clean_sample_indices_B = select_clean_samples_with_autoencoder((data, labels, indices), autoencoder, threshold)

        if epoch < init_epoch:
            loss1, loss2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], indices, noise_or_not)
        else:
            if args.model_type == 'coteaching_plus':
                loss1, loss2, _, _ = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], indices, noise_or_not, epoch * i)
            
            # Only use clean samples for updating the models
            if len(clean_sample_indices_A) > 0 and len(clean_sample_indices_B) > 0:
                clean_sample_indices_A_tensor = torch.tensor(clean_sample_indices_A, dtype=torch.long).cuda()
                clean_sample_indices_B_tensor = torch.tensor(clean_sample_indices_B, dtype=torch.long).cuda()
                
                clean_logits1 = logits1[clean_sample_indices_A_tensor]
                clean_logits2 = logits2[clean_sample_indices_B_tensor]
                clean_labels_A = labels[clean_sample_indices_A_tensor]
                clean_labels_B = labels[clean_sample_indices_B_tensor]
                
                clean_loss1 = F.cross_entropy(clean_logits1, clean_labels_A)
                clean_loss2 = F.cross_entropy(clean_logits2, clean_labels_B)

                loss1 += clean_loss1
                loss2 += clean_loss2

        if args.weight_resampling != 'none':
            weights1 = compute_weights(labels, no_of_classes=no_of_classes)
            weights2 = compute_weights(labels, no_of_classes=no_of_classes)
            loss1 = (loss1 * weights1).mean() 
            loss2 = (loss2 * weights2).mean()

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        _, predicted1 = torch.max(logits1.data, 1)
        _, predicted2 = torch.max(logits2.data, 1)
        train_total += labels.size(0)
        train_correct1 += (predicted1 == labels).sum().item()
        train_correct2 += (predicted2 == labels).sum().item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()

        epoch_losses1.append(loss1.item())
        epoch_losses2.append(loss2.item())

        if (i + 1) % args.print_freq == 0:
            avg_loss1 = total_loss1 / (i + 1)
            avg_loss2 = total_loss2 / (i + 1)
            avg_loss = (avg_loss1 + avg_loss2) / 2
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4F, Loss1: %.4f, Loss2: %.4f, Avg Loss: %.4f' 
                  % (epoch + 1, args.n_epoch, i + 1, len(train_loader), 100. * train_correct1 / train_total, 100. * train_correct2 / train_total, avg_loss1, avg_loss2, avg_loss))

    avg_loss1 = total_loss1 / len(train_loader)
    avg_loss2 = total_loss2 / len(train_loader)
    avg_loss = (avg_loss1 + avg_loss2) / 2

    if hasattr(train, "epoch_losses"):
        train.epoch_losses.append(avg_loss)
        if len(train.epoch_losses) > 10:
            train.epoch_losses.pop(0)
    else:
        train.epoch_losses = [avg_loss]

    avg_loss_last_10_epochs = np.mean(train.epoch_losses)

    print('Epoch [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4F, Avg Loss1: %.4f, Avg Loss2: %.4f, Avg Loss: %.4f, Avg Loss Last 10 Epochs: %.4f' 
          % (epoch + 1, args.n_epoch, 100. * train_correct1 / train_total, 100. * train_correct2 / train_total, avg_loss1, avg_loss2, avg_loss, avg_loss_last_10_epochs))

    train_acc1 = 100. * train_correct1 / train_total
    train_acc2 = 100. * train_correct2 / train_total
    return train_acc1, train_acc2



def introduce_feature_noise(features, label_noise_rate):
    n_samples, n_features = features.shape
    n_noisy = int(label_noise_rate * n_features)

    noisy_features = features.copy()
    noise_indices = np.random.choice(n_features, size=n_noisy, replace=False)

    for i in range(n_samples):
        for j in noise_indices:
            noisy_features[i, j] = np.random.normal(loc=features[i, j], scale=1.0)

    noise_or_not = np.zeros(n_samples, dtype=bool)
    return noisy_features, noise_or_not

def clean_class_name(class_name):
    cleaned_name = re.sub(r'[^a-zA-Z0-9]+', ' ', class_name)
    cleaned_name = re.sub(r'\bweb attack\b', '', cleaned_name, flags=re.IGNORECASE)
    return cleaned_name.strip()

def calculate_metrics(all_labels, all_preds, label_encoder, args):
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_average': np.mean(f1_score(all_labels, all_preds, average=None, zero_division=0))
    }
    return metrics

def create_confusion_matrix(all_labels, all_preds, label_encoder, args):
    colors = ["#FFFFFF", "#B9F5F1", "#C8A8E2"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    if args.dataset == 'CIC_IDS_2017':
        class_names = [label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))]
    elif args.dataset == 'windows_pe_real':
        class_names = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber", "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue", "InstallMonster", "Locky"]
    elif args.dataset == 'BODMAS':
        class_names = [f"Class {i + 1}" for i in range(10)]
    else:
        class_names = [label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))]

    cleaned_class_names = [clean_class_name(name) for name in class_names]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, 
                     xticklabels=cleaned_class_names, 
                     yticklabels=cleaned_class_names, 
                     annot_kws={"fontsize": 14})

    resampling_status = 'weight_resampling' if args.weight_resampling != 'none' else 'no_weight_resampling'
    augmentation = 'No Augmentation' if args.data_augmentation == 'none' else args.data_augmentation.capitalize()
    title = f"{model_str} on {args.dataset.capitalize()} dataset with {augmentation}, {resampling_status.capitalize()}, {args.label_noise_type}-Noise Rate: {args.label_noise_rate}, Imbalance Ratio: {args.imbalance_ratio}"
    
    plt.title(title, fontsize=14, fontweight='bold', loc='left', wrap=True)
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=45, va='center', fontsize=14)
    plt.tight_layout()

    matrix_dir = os.path.join(args.result_dir, 'confusion_matrix')
    os.makedirs(matrix_dir, exist_ok=True)
    matrix_filename = os.path.join(matrix_dir, f"{model_str}_confusion_matrix.png")
    plt.savefig(matrix_filename, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Confusion matrix saved as {matrix_filename}.")

def calculate_class_accuracies(all_labels, all_preds, label_encoder, args):
    if args.dataset == 'CIC_IDS_2017':
        unique_labels = np.unique(all_labels)
        class_accuracy = {f'{label_encoder.inverse_transform([label])[0]}_acc': np.mean([all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label]) for label in unique_labels}
    elif args.dataset == 'windows_pe_real':
        index_to_class_name = {
            i: name for i, name in enumerate([
                "Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                "InstallMonster", "Locky"])
        }
        unique_labels = np.unique(all_labels)
        class_accuracy = {
            f'{index_to_class_name[label]}_acc': np.mean([
                all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label
            ]) for label in unique_labels
        }
    elif args.dataset == 'BODMAS':
        index_to_class_name = {
            i: f"Class {i + 1}" for i in range(len(label_encoder.classes_))
        }
        unique_labels = np.unique(all_labels)
        class_accuracy = {
            f'{index_to_class_name[label]}_acc': np.mean([
                all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label
            ]) for label in unique_labels
        }
    else:
        class_accuracy = {}

    return class_accuracy

def evaluate(test_loader, model1, model2, label_encoder, args, save_conf_matrix=False, return_predictions=False):
    model1.eval()
    model2.eval()
    all_preds1 = []
    all_preds2 = []
    all_labels = []

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.cuda()
            labels = labels.cuda()
            outputs1 = model1(data)
            outputs2 = model2(data)
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            all_preds1.extend(preds1.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if return_predictions:
        return all_preds1, all_preds2
    
    metrics1 = calculate_metrics(all_labels, all_preds1, label_encoder, args)
    metrics2 = calculate_metrics(all_labels, all_preds2, label_encoder, args)

    class_accuracies1 = calculate_class_accuracies(all_labels, all_preds1, label_encoder, args)
    class_accuracies2 = calculate_class_accuracies(all_labels, all_preds2, label_encoder, args)

    metrics1.update(class_accuracies1)
    metrics2.update(class_accuracies2)
    
    if save_conf_matrix:
        create_confusion_matrix(all_labels, all_preds1, label_encoder, args)
        create_confusion_matrix(all_labels, all_preds2, label_encoder, args)

    return metrics1, metrics2

def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

def handle_inf_nan(features_np):
    print("Contains inf: ", np.isinf(features_np).any())
    print("Contains -inf: ", np.isneginf(features_np).any())
    print("Contains NaN: ", np.isnan(features_np).any())
    features_np[np.isinf(features_np) | np.isneginf(features_np)] = np.nan
    imputer = SimpleImputer(strategy='median')
    features_np = imputer.fit_transform(features_np)
    scaler = StandardScaler()
    return scaler.fit_transform(features_np)

def main():
    print(model_str)
    print(model_str)
    print(model_str)
    label_encoder = LabelEncoder()

    if args.dataset == 'CIC_IDS_2017':
        preprocessed_file_path = 'data/final_dataframe.csv'
        if not os.path.exists(preprocessed_file_path):
            filenames = [os.path.join("data/cicids2017/MachineLearningCSV/MachineLearningCVE", f) for f in os.listdir("data/cicids2017/MachineLearningCSV/MachineLearningCVE") if f.endswith('.csv')]
            df_list = [pd.read_csv(filename) for filename in filenames]
            df = pd.concat(df_list, ignore_index=True)
            df.columns = df.columns.str.strip()
            df.to_csv(preprocessed_file_path, index=False)
            print("Saved concatenated data")
        df = pd.read_csv(preprocessed_file_path)
        labels_np = label_encoder.fit_transform(df['Label'].values)
        features_np = df.drop('Label', axis=1).values.astype(np.float32)
        features_np = handle_inf_nan(features_np)

        X_train, X_temp, y_train, y_temp = train_test_split(features_np, labels_np, test_size=0.4, random_state=42)
        X_test, X_clean_test, y_test, y_clean_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    elif args.dataset == 'windows_pe_real':
        npz_noisy_file_path = 'data/Windows_PE/real_world/malware.npz'
        npz_clean_file_path = 'data/Windows_PE/real_world/malware_true.npz'
        with np.load(npz_noisy_file_path) as data:
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        with np.load(npz_clean_file_path) as data:
            X_clean_test, y_clean_test = data['X_test'], data['y_test']
        y_clean_test = label_encoder.transform(y_clean_test)

    elif args.dataset == 'BODMAS':
        npz_file_path = 'data/Windows_PE/synthetic/malware_true.npz'
        with np.load(npz_file_path) as data:
            X_temp, y_temp = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_temp = label_encoder.fit_transform(y_temp)
        y_test = label_encoder.transform(y_test)
        
        X_train, X_clean_test, y_train, y_clean_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    results_dir = os.path.join(args.result_dir, args.dataset, args.model_type)
    os.makedirs(results_dir, exist_ok=True)

    resampling_status = 'weight_resampling' if args.weight_resampling else 'no_weight_resampling'
    if args.weight_resampling != 'none':
        base_filename = f"{args.model_type}_{args.dataset}_dataset_{args.data_augmentation if args.data_augmentation != 'none' else 'no_augmentation'}_{args.weight_resampling}_{resampling_status}_{args.label_noise_type}-noise{args.label_noise_rate}_imbalance{args.imbalance_ratio}"
    else:
        base_filename = f"{args.model_type}_{args.dataset}_dataset_{args.data_augmentation if args.data_augmentation != 'none' else 'no_augmentation'}_{resampling_status}_{args.label_noise_type}-noise{args.label_noise_rate}_imbalance{args.imbalance_ratio}"

    validation_metrics_file_model1 = os.path.join(results_dir, f"{base_filename}_validation_metrics_model1.csv")
    validation_metrics_file_model2 = os.path.join(results_dir, f"{base_filename}_validation_metrics_model2.csv")
    full_dataset_metrics_file_model1 = os.path.join(results_dir, f"{base_filename}_full_dataset_model1.csv")
    full_dataset_metrics_file_model2 = os.path.join(results_dir, f"{base_filename}_full_dataset_model2.csv")

    final_model_path = os.path.join(results_dir, f"{base_filename}_final_model.pth")

    with open(validation_metrics_file_model1, "w", newline='', encoding='utf-8') as csvfile:
        if args.dataset == 'BODMAS':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'Class {label+1}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'CIC_IDS_2017':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'windows_pe_real':
            labels = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                      "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                      "InstallMonster", "Locky"]
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in labels]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with open(validation_metrics_file_model2, "w", newline='', encoding='utf-8') as csvfile:
        if args.dataset == 'BODMAS':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'Class {label+1}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'CIC_IDS_2017':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'windows_pe_real':
            labels = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                      "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                      "InstallMonster", "Locky"]
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in labels]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with open(full_dataset_metrics_file_model1, "w", newline='', encoding='utf-8') as csvfile:
        if args.dataset == 'BODMAS':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'Class {label+1}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'CIC_IDS_2017':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'windows_pe_real':
            labels = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                      "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                      "InstallMonster", "Locky"]
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in labels]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with open(full_dataset_metrics_file_model2, "w", newline='', encoding='utf-8') as csvfile:
        if args.dataset == 'BODMAS':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'Class {label+1}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'CIC_IDS_2017':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'windows_pe_real':
            labels = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                      "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                      "InstallMonster", "Locky"]
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro', 'f1_average'] + \
                         [f'{label}_acc' for label in labels]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    print("Original dataset:")
    print(f"Length of X_train: {len(X_train)}")
    print(f"Length of y_train: {len(y_train)}")
    print("Class distribution in original dataset:", {label: np.sum(y_train == label) for label in np.unique(y_train)})

    X_train_imbalanced, y_train_imbalanced = apply_imbalance(X_train, y_train, args.imbalance_ratio)
    print("Before introducing noise:")
    print(f"Length of X_train_imbalanced: {len(X_train_imbalanced)}")
    print(f"Length of y_train_imbalanced: {len(y_train_imbalanced)}")
    print("Class distribution after applying imbalance:", {label: np.sum(y_train_imbalanced == label) for label in np.unique(y_train_imbalanced)})

    # Apply feature noise to the imbalanced data
    X_train_imbalanced = feature_noise(torch.tensor(X_train_imbalanced), add_noise_level=args.feature_add_noise_level, mult_noise_level=args.feature_mult_noise_level).numpy()

    # Introduce noise to the imbalanced data
    y_train__label_noisy, label_noise_or_not = introduce_noise(y_train_imbalanced, X_train_imbalanced, args.label_noise_type, args.label_noise_rate)

    # Print class distribution after introducing noise
    print("Before augmentation:")
    print(f"Length of X_train_imbalanced: {len(X_train_imbalanced)}")
    print(f"Length of y_train__label_noisy: {len(y_train__label_noisy)}")
    print(f"Length of label_noise_or_not: {len(label_noise_or_not)}")
    print("Class distribution after introducing noise:", {label: np.sum(y_train__label_noisy == label) for label in np.unique(y_train__label_noisy)})

    X_train_augmented, y_train_augmented = apply_data_augmentation(X_train_imbalanced, y_train_imbalanced, args.data_augmentation)
    if args.data_augmentation in ['smote', 'adasyn', 'oversampling']:
        noise_or_not = np.zeros(len(y_train_augmented), dtype=bool)

    print("After augmentation:")
    print(f"Length of X_train_augmented: {len(X_train_augmented)}")
    print(f"Length of y_train_augmented: {len(y_train_augmented)}")
    print(f"Length of noise_or_not (adjusted if necessary): {len(label_noise_or_not)}")
    print("Class distribution after data augmentation:", {label: np.sum(y_train_augmented == label) for label in np.unique(y_train_augmented)})

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    results = []
    fold = 0
    for train_idx, val_idx in skf.split(X_train_augmented, y_train_augmented):
        fold += 1
        X_train_fold, X_val_fold = X_train_augmented[train_idx], X_train_augmented[val_idx]
        y_train_fold, y_val_fold = y_train_augmented[train_idx], y_train_augmented[val_idx]
        noise_or_not_train, noise_or_not_val = label_noise_or_not[train_idx], label_noise_or_not[val_idx]

        train_dataset = CICIDSDataset(X_train_fold, y_train_fold, noise_or_not_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        val_dataset = CICIDSDataset(X_clean_test, y_clean_test, np.zeros(len(y_clean_test), dtype=bool))
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        model1 = MLPNet(num_features=X_train_fold.shape[1], num_classes=len(np.unique(y_train_fold)), dataset=args.dataset).cuda()
        model2 = MLPNet(num_features=X_train_fold.shape[1], num_classes=len(np.unique(y_train_fold)), dataset=args.dataset).cuda()
        model1.apply(weights_init)
        model2.apply(weights_init)
        optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
        optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)

        autoencoder = Autoencoder(input_dim=X_train_augmented.shape[1]).cuda()
        autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr)
        for epoch in range(args.n_epoch):
            autoencoder.train()
            for data, _, _ in train_loader:
                data = data.cuda()
                recon_data = autoencoder(data)
                loss = F.mse_loss(recon_data, data)
                autoencoder_optimizer.zero_grad()
                loss.backward()
                autoencoder_optimizer.step()

        threshold = np.percentile(compute_reconstruction_error(autoencoder, train_loader), 95)

        for epoch in range(args.n_epoch):
            no_of_classes = len(np.unique(y_train))
            train(train_loader, model1, optimizer1, model2, optimizer2, epoch, args, no_of_classes, noise_or_not_train, autoencoder, threshold)

            metrics1, metrics2 = evaluate(val_loader, model1, model2, label_encoder, args, save_conf_matrix=False)

            row_data1 = OrderedDict([('Fold', fold), ('Epoch', epoch)] + list(metrics1.items()))
            row_data2 = OrderedDict([('Fold', fold), ('Epoch', epoch)] + list(metrics2.items()))
            with open(validation_metrics_file_model1, "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data1)
            with open(validation_metrics_file_model2, "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data2)
        results.append((metrics1, metrics2))

    print("Training completed. Results from all folds:")
    for i, (res1, res2) in enumerate(results, 1):
        print(f'Results Fold {i}: Model 1: {res1}, Model 2: {res2}')

    print("Training on the full dataset...")
    full_train_dataset = CICIDSDataset(X_train_augmented, y_train_augmented, label_noise_or_not)
    full_train_loader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    full_model1 = MLPNet(num_features=X_train_augmented.shape[1], num_classes=len(np.unique(y_train_augmented)), dataset=args.dataset).cuda()
    full_model1.apply(weights_init)
    full_optimizer1 = optim.Adam(full_model1.parameters(), lr=args.lr)

    full_model2 = MLPNet(num_features=X_train_augmented.shape[1], num_classes=len(np.unique(y_train_augmented)), dataset=args.dataset).cuda()
    full_model2.apply(weights_init)
    full_optimizer2 = optim.Adam(full_model2.parameters(), lr=args.lr)

    for epoch in range(args.n_epoch):
        no_of_classes = len(np.unique(y_train))
        train(full_train_loader, full_model1, full_optimizer1, full_model2, full_optimizer2, epoch, args, no_of_classes, label_noise_or_not, autoencoder, threshold)

    clean_test_dataset = CICIDSDataset(X_clean_test, y_clean_test, np.zeros_like(y_clean_test, dtype=bool))
    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    full_metrics1, full_metrics2 = evaluate(clean_test_loader, full_model1, full_model2, label_encoder, args, save_conf_matrix=True)
    predictions1, predictions2 = evaluate(clean_test_loader, full_model1, full_model2, label_encoder, args, return_predictions=True)

    predictions_dir = os.path.join(results_dir, args.dataset, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    predictions_filename1 = os.path.join(predictions_dir, f"{base_filename}_predictions_model1.csv")
    predictions_filename2 = os.path.join(predictions_dir, f"{base_filename}_predictions_model2.csv")
    
    save_predictions(predictions1, predictions_filename1)
    save_predictions(predictions2, predictions_filename2)

    print(full_metrics1)
    record_evaluation_results(full_metrics1, full_dataset_metrics_file_model1, epoch, fieldnames)
    record_evaluation_results(full_metrics2, full_dataset_metrics_file_model2, epoch, fieldnames)

    print("Final evaluation completed.")

def save_predictions(predictions, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Predicted Label'])
        for pred in predictions:
            writer.writerow([pred])

def record_evaluation_results(metrics, filename, epoch, fieldnames):
    if not isinstance(metrics, dict):
        raise ValueError("Metrics should be a dictionary.")
    row_data = OrderedDict([('Epoch', epoch)] + list(metrics.items()))
    with open(filename, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.write(row_data)

if __name__ == '__main__':
    main()
