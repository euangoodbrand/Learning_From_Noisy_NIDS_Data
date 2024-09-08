# -*- coding:utf-8 -*-
from __future__ import print_function 
from model import MLPNet

# General Imports
import os
import re
import datetime
import argparse

# Maths and Data processing imports
import numpy as np
import pandas as pd
import random

# Plotting Imports
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Sklearn Import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# sklearn imports
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError

# Imblearn Imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, ADASYN

# Pytorch Imports
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None 

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.0)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='Type of noise to introduce', choices=['uniform', 'class', 'feature','MIMICRY'], default='uniform')
parser.add_argument('--num_gradual', type=int, default=5, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type=str, help='cicids', choices=['CIC_IDS_2017','windows_pe_real','BODMAS', 'Android_multiclass'])
parser.add_argument('--n_epoch', type=int, default=15)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus]', default='baseline')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--data_augmentation', type=str, choices=['none', 'smote', 'undersampling', 'oversampling', 'adasyn'], default='none', help='Data augmentation technique, if any')
parser.add_argument('--imbalance_ratio', type=float, default=0.0, help='Ratio to imbalance the dataset')
parser.add_argument('--weight_resampling', type=str, choices=['none','Naive', 'Focal', 'Class-Balance'], default='none', help='Select the weight resampling method if needed')
parser.add_argument('--feature_add_noise_level', type=float, default=0.0, help='Level of additive noise for features')
parser.add_argument('--feature_mult_noise_level', type=float, default=0.0, help='Level of multiplicative noise for features')
parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization weight decay. Default is 0 (no regularization).')
args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

# Hyper Parameters


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
elif args.dataset == "Android_multiclass":
    batch_size = 256
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
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

def introduce_noise(labels, features, noise_type, noise_rate):
    if noise_type == 'uniform':
        return introduce_uniform_noise(labels, noise_rate)
    elif noise_type == 'class':
        # Directly use the predefined matrix for class noise
        return introduce_class_dependent_label_noise(labels, predefined_matrix, noise_rate)
    elif noise_type == 'feature':
        thresholds = calculate_feature_thresholds(features)
        return introduce_feature_dependent_label_noise(features, labels, noise_rate, n_neighbors=5)
    elif noise_type == 'MIMICRY':
        # Directly use the predefined matrix for class noise
        return introduce_mimicry_noise(labels, predefined_matrix, noise_rate)
    else:
        raise ValueError("Invalid noise type specified.")

def apply_data_augmentation(features, labels, augmentation_method):
    try:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class distribution before augmentation: {dict(zip(unique, counts))}")

        if augmentation_method == 'smote':
            # Adjust n_neighbors based on the smallest class count minus one (since it cannot be more than the number of samples in the smallest class)
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

# Class dependant noise matrix, from previous evaluation run.
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

# MIMICRY matrix for windows pe created from windows clean and noisy data
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

def introduce_class_dependent_label_noise(labels, class_noise_matrix, noise_rate):
    if noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)  # Return the original labels with no noise

    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    new_labels = labels.copy()
    noise_or_not = np.zeros(n_samples, dtype=bool)

    for idx in noisy_indices:
        original_class = labels[idx]
        new_labels[idx] = np.random.choice(np.arange(len(class_noise_matrix[original_class])), p=class_noise_matrix[original_class])
        noise_or_not[idx] = new_labels[idx] != labels[idx]

    return new_labels, noise_or_not

def introduce_mimicry_noise(labels, class_noise_matrix, noise_rate):
    if noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    new_labels = labels.copy()
    noise_or_not = np.zeros(n_samples, dtype=bool)

    for idx in noisy_indices:
        original_class = labels[idx]
        new_labels[idx] = np.random.choice(np.arange(len(class_noise_matrix[original_class])), p=class_noise_matrix[original_class])
        noise_or_not[idx] = new_labels[idx] != labels[idx]

    return new_labels, noise_or_not

def calculate_feature_thresholds(features):
    # Calculate thresholds for each feature, assuming features is a 2D array
    thresholds = np.percentile(features, 50, axis=0)  # Median as threshold for each feature
    return thresholds

def introduce_feature_dependent_label_noise(features, labels, noise_rate, n_neighbors=5):
    if noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn.fit(features)
    distances, indices = knn.kneighbors(features)

    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)
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

def introduce_uniform_noise(labels, noise_rate):
    if noise_rate == 0:
        return labels.copy(), np.zeros(len(labels), dtype=bool)

    n_samples = len(labels)
    n_noisy = int(noise_rate * n_samples)
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

    # Identify the unique classes and their counts
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    
    # Determine which classes to downsample
    if downsample_half:
        downsample_classes = unique[n_classes // 2:]
        keep_classes = unique[:n_classes // 2]
    else:
        downsample_classes = unique
        keep_classes = []

    # Calculate the average count of the classes not being downsampled
    keep_indices = []
    keep_class_counts = [count for cls, count in zip(unique, counts) if cls in keep_classes]
    if keep_class_counts:
        average_keep_class_count = int(np.mean(keep_class_counts))
    else:
        average_keep_class_count = min(counts)  # Fallback if no classes are kept
    
    indices_to_keep = []
    for cls in unique:
        class_indices = np.where(labels == cls)[0]
        if cls in downsample_classes:
            # Calculate the target count for downsampled classes
            n_majority_new = max(int(average_keep_class_count * ratio), min_samples_per_class)
            if len(class_indices) > n_majority_new:
                keep_indices = np.random.choice(class_indices, n_majority_new, replace=False)
            else:
                keep_indices = class_indices  # Keep all samples if class count is below the target
        else:
            # Keep all samples from the classes not being downsampled
            keep_indices = class_indices

        indices_to_keep.extend(keep_indices)

    indices_to_keep = np.array(indices_to_keep)
    np.random.shuffle(indices_to_keep)  # Shuffle indices to mix classes
    
    return features[indices_to_keep], labels[indices_to_keep]


def compute_weights(labels, no_of_classes, beta=0.9999, gamma=2.0, device='cuda'):
    # Convert labels to a numpy array if it's a tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Count each class's occurrence
    samples_per_class = np.bincount(labels, minlength=no_of_classes)

    # Handling different weight resampling strategies
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

    # Normalize weights to sum to number of classes
    total_weight = np.sum(weights)
    if (total_weight == 0):
        weights = np.ones(no_of_classes, dtype=np.float32)
    else:
        weights = (weights / total_weight) * no_of_classes

    # Convert numpy weights to torch tensor and move to the specified device
    weight_per_label = torch.from_numpy(weights).float().to(device)

    # Index weights by labels
    weight_per_label = weight_per_label[torch.from_numpy(labels).to(device)]
    
    return weight_per_label
  
save_dir = args.result_dir +'/' +args.dataset+'/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

# Define model string including sample reweighting
model_str = (
    f"{args.model_type}_{args.dataset}_"
    f"{'no_augmentation' if args.data_augmentation == 'none' else args.data_augmentation}_"
    f"{args.noise_type}-noise{args.noise_rate}_imbalance{args.imbalance_ratio}_"
    f"addNoise{args.feature_add_noise_level}_multNoise{args.feature_mult_noise_level}_"
    f"{args.weight_resampling if args.weight_resampling != 'none' else 'no_weight_resampling'}"
)

txtfile = save_dir + "/" + model_str + ".csv"
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile + ".bak-%s" % nowTime))

def train(train_loader, model, optimizer, criterion, epoch, no_of_classes):
    model.train()  # Set model to training mode
    train_total = 0
    train_correct = 0
    total_loss = 0

    for i, (data, labels, _) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        logits = model(data)

        # Compute the standard CrossEntropyLoss
        loss = criterion(logits, labels)
        
        # Apply weights manually if weight resampling is enabled
        if args.weight_resampling != 'none':
            weights = compute_weights(labels, no_of_classes=no_of_classes)
            loss = (loss * weights).mean() 

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        total_loss += loss.item()

        # if (i + 1) % args.print_freq == 0:
        #     print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
        #           % (epoch + 1, args.n_epoch, i + 1, len(train_loader), 100. * train_correct / train_total, total_loss / train_total))

    print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch + 1, args.n_epoch, i + 1, len(train_loader), 100. * train_correct / train_total, total_loss / train_total))

    train_acc = 100. * train_correct / train_total
    return train_acc

def clean_class_name(class_name):
    # Replace non-standard characters with spaces
    cleaned_name = re.sub(r'[^a-zA-Z0-9]+', ' ', class_name)
    # Remove "web attack" in any case combination
    cleaned_name = re.sub(r'\bweb attack\b', '', cleaned_name, flags=re.IGNORECASE)
    return cleaned_name.strip()  # Strip leading/trailing spaces

def evaluate(test_loader, model, label_encoder, args, save_conf_matrix=False, return_predictions=False):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data = data.cuda()
            labels = labels.cuda()
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if return_predictions:
        return all_preds

    if args.dataset == 'CIC_IDS_2017':
            index_to_class_name = {i: label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))}
    elif args.dataset == 'windows_pe_real':
        index_to_class_name = {
            i: name for i, name in enumerate([
                "Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                "InstallMonster", "Locky"])
        }
    elif args.dataset == 'BODMAS':
        index_to_class_name = {
            i: name for i, name in enumerate([
                "Class 1", "Class 2", "Class 3", "Class 4", "Class 5",
                "Class 6", "Class 7", "Class 8", "Class 9", "Class 10"])
        }
    elif args.dataset == 'Android_multiclass':
        index_to_class_name = {i: label_encoder.classes_[i] for i in range(len(label_encoder.classes_))}
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
            
    cleaned_class_names = [clean_class_name(str(name)) for name in index_to_class_name.values()]
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_average': np.mean(f1_score(all_labels, all_preds, average=None, zero_division=0))  # Average F1 score
    }

    # Class accuracy
    unique_labels = np.unique(all_labels)
    if args.dataset == 'CIC_IDS_2017':
        class_accuracy = {f'{label_encoder.inverse_transform([label])[0]}_acc': np.mean([all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label]) for label in unique_labels}
    elif args.dataset == 'windows_pe_real':
        class_accuracy = {
            f'{index_to_class_name[label]}_acc': np.mean([
                all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label
            ]) for label in unique_labels
        }
    elif args.dataset == 'BODMAS':
        class_accuracy = {
            f'{index_to_class_name[label]}_acc': np.mean([
                all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label
            ]) for label in unique_labels
        }
    elif args.dataset == 'Android_multiclass':
        class_accuracy = {
            f'{index_to_class_name[label]}_acc': np.mean([
                all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label
            ]) for label in unique_labels
        }
    else:
        class_accuracy = {}  # Empty dict if dataset is not recognized

    metrics.update(class_accuracy)

    if save_conf_matrix:

        # Define colors
        colors = ["#FFFFFF", "#B9F5F1", "#C8A8E2"]  
        # Create a color map
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)    
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        print(cm)
        plt.figure(figsize=(12, 10))

        resampling_status = 'weight_resampling' if args.weight_resampling != 'none' else 'no_weight_resampling'

        if args.weight_resampling != 'none':
            title = (f"{args.model_type.capitalize()} on {args.dataset.capitalize()} dataset with "
            f"{'No Augmentation' if args.data_augmentation == 'none' else args.data_augmentation.capitalize()},\n"
            f"{args.weight_resampling}_{resampling_status.capitalize()},"
            f"{args.noise_type}-Noise Rate: {args.noise_rate}, Imbalance Ratio: {args.imbalance_ratio}"
            )
        else: 
            title = (f"{args.model_type.capitalize()} on {args.dataset.capitalize()} dataset with "
            f"{'No Augmentation' if args.data_augmentation == 'none' else args.data_augmentation.capitalize()},\n"
            f"{resampling_status.capitalize()},"
            f"{args.noise_type}-Noise Rate: {args.noise_rate}, Imbalance Ratio: {args.imbalance_ratio}"
            )
        ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=cleaned_class_names, yticklabels=cleaned_class_names, annot_kws={"fontsize": 14})    

        # Generate custom tick labels (arrow symbol) for the number of classes
        tick_labels1 = [f"{name}" for name in cleaned_class_names]    
        tick_labels2 = [f"{name}" for name in cleaned_class_names]  

        
        ax.set_xticklabels(tick_labels1, rotation=90)  # Set rotation for x-axis labels
        ax.set_yticklabels(tick_labels2, rotation=0)  # Align y-axis labels
        ax.tick_params(left=False, bottom=False, pad=10)


        # Define Unicode characters for ticks
        unicode_symbol_x = chr(0x25B6)  # Black right-pointing triangle
        unicode_symbol_y = chr(0x25B2)  # Black up-pointing triangle

        # Get current tick locations
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # Disable original ticks
        ax.tick_params(length=0)  # Hide tick lines

    
        for y in yticks:
            ax.annotate(unicode_symbol_x, xy=(0, y), xycoords='data', xytext=(0, 0), textcoords='offset points', ha='right', va='center', fontsize=12, color='black')
        # Overlay Unicode characters as custom tick marks
        for x in xticks:
            ax.annotate(unicode_symbol_y, xy=(x, y+0.5), xycoords='data', xytext=(0, 0), textcoords='offset points', ha='center', va='top', fontsize=12, color='black')

        plt.xticks(rotation=45, ha='right', fontsize=14)  
        plt.yticks(rotation=45, va='top', fontsize=14)
        plt.xlabel('Predicted', fontsize=14, fontweight='bold')  
        plt.ylabel('Actual', fontsize=14, fontweight='bold') 
        plt.title(title, fontsize=14, fontweight='bold', loc='left', wrap=True)  
        plt.tight_layout()

        # Adding border around the color bar
        cbar = plt.gca().collections[0].colorbar
        cbar.outline.set_linewidth(1)
        cbar.outline.set_edgecolor("black")

        matrix_dir = os.path.join(args.result_dir, 'confusion_matrix')
        if not os.path.exists(matrix_dir):
            os.makedirs(matrix_dir)
        
        matrix_filename = f"{model_str}_confusion_matrix.png"
        plt.savefig(os.path.join(matrix_dir, matrix_filename), bbox_inches='tight', dpi=300)
        plt.close()

    return metrics

def weights_init(m):
    if isinstance(m, nn.Linear):
        # Apply custom initialization to linear layers
        init.xavier_uniform_(m.weight.data)  # Xavier initialization for linear layers
        if m.bias is not None:
            init.constant_(m.bias.data, 0)    # Initialize bias to zero

    elif isinstance(m, nn.Conv2d):
        # Apply custom initialization to convolutional layers
        init.kaiming_normal_(m.weight.data)   # Kaiming initialization for convolutional layers
        if m.bias is not None:
            init.constant_(m.bias.data, 0)     # Initialize bias to zero

def handle_inf_nan(features_np):
    print("Contains inf: ", np.isinf(features_np).any())
    print("Contains -inf: ", np.isneginf(features_np).any())
    print("Contains NaN: ", np.isnan(features_np).any())
    features_np[np.isinf(features_np) | np.isneginf(features_np)] = np.nan
    imputer = SimpleImputer(strategy='median')
    features_np = imputer.fit_transform(features_np)
    scaler = StandardScaler()
    return scaler.fit_transform(features_np)

def train_single_model(model, train_loader, val_loader, args, label_encoder, epochs=150):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = None

    for epoch in range(epochs):
        model.train()
        for i, (data, labels, _) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data, labels = data.cuda(), labels.cuda()
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        print(f'Epoch [{epoch+1}/{epochs}], Validation Accuracy: {val_acc:.2f}%')

    return best_model

def iterative_noisy_student_training(X_clean, y_clean, X_removed, y_removed, X_true_clean, y_true_clean, X_val, y_val, label_encoder, num_classes, args, num_iterations=3):
    # Create datasets and loaders
    clean_dataset = CICIDSDataset(X_clean, y_clean, np.zeros_like(y_clean, dtype=bool))
    clean_loader = DataLoader(dataset=clean_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_dataset = CICIDSDataset(X_val, y_val, np.zeros_like(y_val, dtype=bool))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    
    true_clean_dataset = CICIDSDataset(X_true_clean, y_true_clean, np.zeros_like(y_true_clean, dtype=bool))
    true_clean_loader = DataLoader(dataset=true_clean_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # Train initial teacher model
    print("Training initial teacher model on clean dataset...")
    teacher_model = create_model(X_clean.shape[1], num_classes, args.dataset)
    teacher_model = train_single_model(teacher_model, clean_loader, val_loader, args, label_encoder)

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Generate pseudo-labels for removed samples
        print("Generating pseudo-labels for removed samples...")
        teacher_model.eval()
        pseudo_labels = []
        with torch.no_grad():
            for batch in DataLoader(X_removed, batch_size=batch_size):
                batch = batch.cuda()
                outputs = teacher_model(batch)
                _, predicted = torch.max(outputs, 1)
                pseudo_labels.extend(predicted.cpu().numpy())

        # Combine clean and pseudo-labeled removed data
        X_combined = np.vstack((X_clean, X_removed))
        y_combined = np.concatenate((y_clean, pseudo_labels))
        combined_dataset = CICIDSDataset(X_combined, y_combined, np.zeros_like(y_combined, dtype=bool))
        combined_loader = DataLoader(dataset=combined_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        # Train student model
        print("Training student model on combined dataset...")
        student_model = create_model(X_combined.shape[1], num_classes, args.dataset, dropout_rate=0.5)
        student_model = train_single_model(student_model, combined_loader, val_loader, args, label_encoder)

        # Evaluate models
        print("Evaluating models on true clean test set...")
        teacher_metrics = evaluate(true_clean_loader, teacher_model, label_encoder, args)
        student_metrics = evaluate(true_clean_loader, student_model, label_encoder, args)

        print("\nTeacher Model Metrics:")
        for key, value in teacher_metrics.items():
            print(f"{key}: {value}")

        print("\nStudent Model Metrics:")
        for key, value in student_metrics.items():
            print(f"{key}: {value}")

        # Set the student as the new teacher for the next iteration
        teacher_model = student_model

    return teacher_metrics, student_metrics

def train_single_mlp(X_train, y_train, noise_or_not, X_val, y_val, args, label_encoder, num_classes, epochs=15):
    train_dataset = CICIDSDataset(X_train, y_train, noise_or_not)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_dataset = CICIDSDataset(X_val, y_val, np.zeros(len(y_val), dtype=bool))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = MLPNet(num_features=X_train.shape[1], num_classes=num_classes, dataset=args.dataset).cuda()
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model = None

    for epoch in range(epochs):
        train(train_loader, model, optimizer, criterion, epoch, len(np.unique(y_train)))
        metrics = evaluate(val_loader, model, label_encoder, args, save_conf_matrix=False)
        
        if metrics['accuracy'] > best_val_acc:
            best_val_acc = metrics['accuracy']
            best_model = copy.deepcopy(model)

    return best_model

def ensemble_predict(models, X):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X, dtype=torch.float32).cuda())
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.cpu().numpy())
    
    predictions = np.array(predictions)

    # Majority voting
    ensemble_preds = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

    # Find indexes where all predictions agree
    all_agree_indexes = np.where(np.all(predictions == predictions[0, :], axis=0))[0]

    # Find indexes where exactly four predictions agree
    exactly_four_agree_indexes = []
    exactly_three_agree_indexes = []
    for i in range(predictions.shape[1]):  # iterate over each sample
        unique, counts = np.unique(predictions[:, i], return_counts=True)
        if 6 in counts:
            exactly_four_agree_indexes.append(i)
        elif 5 in counts: 
            exactly_three_agree_indexes.append(i)
    exactly_four_agree_indexes = np.array(exactly_four_agree_indexes)

    return ensemble_preds, all_agree_indexes, exactly_four_agree_indexes, exactly_three_agree_indexes


def train_ensemble(X_train, y_train, X_val, y_val, label_encoder, num_classes, num_models=7):
    ensemble_models = []
    for i in range(num_models):
        print(f"\nTraining MLP model {i+1}/{num_models}")
        model = train_single_mlp(X_train, y_train, np.zeros_like(y_train, dtype=bool), X_val, y_val, args, label_encoder, num_classes, epochs=15)
        ensemble_models.append(model)
    return ensemble_models


def create_ensemble_clean_dataset(ensemble_models, X, y):
    ensemble_predictions, all_agree_indexes, _, _ = ensemble_predict(ensemble_models, X)
    clean_indices = all_agree_indexes[ensemble_predictions[all_agree_indexes] == y[all_agree_indexes]]
    return X[clean_indices], y[clean_indices]

def create_model(num_features, num_classes, dataset, dropout_rate=0.0):
    return MLPNet(num_features=num_features, num_classes=num_classes, dataset=dataset, dropout_rate=dropout_rate).cuda()

def main():

    label_encoder = LabelEncoder()

    # Load and preprocess data based on the dataset
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
        with np.load(npz_clean_file_path) as data:
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
        with np.load(npz_clean_file_path) as data:
            X_clean_test, y_clean_test = data['X_test'], data['y_test']
        y_clean_test = label_encoder.transform(y_clean_test)
    elif args.dataset == 'BODMAS':
        npz_file_path = 'data/Windows_PE/synthetic/malware.npz'
        with np.load(npz_file_path) as data:
            X_temp, y_temp = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_temp = label_encoder.fit_transform(y_temp)
        y_test = label_encoder.transform(y_test)
        
        X_train, X_clean_test, y_train, y_clean_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
    elif args.dataset =='Android_multiclass':
        npz_file_path = 'data/android/multi_class.npz'
        with np.load(npz_file_path) as data:
            X_data, y_data = data['x_data'], data['y_data']
        X_train, X_clean_test, y_train, y_clean_test = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
        y_train = label_encoder.fit_transform(y_train)
        y_clean_test = label_encoder.transform(y_clean_test)
    elif args.dataset =='Android_binary':
        npz_file_path = 'data/android/APIGraphBinary.npz'
        with np.load(npz_file_path) as data:
            X_data, y_data = data['X_train'], data['y_train_binary']
        X_train, X_clean_test, y_train, y_clean_test = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
        y_train = label_encoder.fit_transform(y_train)
        y_clean_test = label_encoder.transform(y_clean_test)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Print original dataset information
    print("Original dataset:")
    print(f"Length of X_train: {len(X_train)}")
    print(f"Length of y_train: {len(y_train)}")
    print("Class distribution in original dataset:", {label: np.sum(y_train == label) for label in np.unique(y_train)})

    # Apply imbalance
    X_train_imbalanced, y_train_imbalanced = apply_imbalance(X_train, y_train, args.imbalance_ratio)
    
    # Introduce noise
    y_train_noisy, noise_or_not = introduce_noise(y_train_imbalanced, X_train_imbalanced, args.noise_type, args.noise_rate)
    
    # Apply feature noise
    X_train_noisy = feature_noise(torch.tensor(X_train_imbalanced), 
                                  add_noise_level=args.feature_add_noise_level, 
                                  mult_noise_level=args.feature_mult_noise_level).numpy()

    # Apply data augmentation
    X_train_augmented, y_train_augmented = apply_data_augmentation(X_train_noisy, y_train_noisy, args.data_augmentation)

    # Train ensemble for cleaning
    print("\nTraining ensemble for data cleaning...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_augmented, y_train_augmented, test_size=0.2, random_state=42)
    num_classes = len(np.unique(y_train_final))
    cleaning_ensemble = train_ensemble(X_train_final, y_train_final, X_val, y_val, label_encoder, num_classes=num_classes)

    # Create ensemble_clean dataset
    print("\nCreating ensemble_clean dataset...")
    X_ensemble_clean, y_ensemble_clean = create_ensemble_clean_dataset(cleaning_ensemble, X_train_augmented, y_train_augmented)
    y_ensemble_clean = label_encoder.fit_transform(y_ensemble_clean)  # Re-encode labels

    # Identify samples removed during cleaning
    removed_indices = np.ones(len(X_train_augmented), dtype=bool)
    removed_indices[np.isin(X_train_augmented, X_ensemble_clean).all(axis=1)] = False
    X_removed = X_train_augmented[removed_indices]
    y_removed = y_train_augmented[removed_indices]

    # Define datasets for training
    datasets = {
        'removed': (X_removed, y_removed),
        'ensemble_clean': (X_ensemble_clean, y_ensemble_clean),
        'clean': (X_train, y_train)
    }

    # Define datasets for Noisy Student Training
    X_clean, y_clean = datasets['ensemble_clean']
    X_removed, y_removed = datasets['removed']
    X_true_clean, y_true_clean = datasets['clean']

    # Split validation set from the clean data
    X_clean_train, X_val, y_clean_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Perform Iterative Noisy Student Training
    num_classes = len(np.unique(y_clean))
    teacher_metrics, student_metrics = iterative_noisy_student_training(
        X_clean_train, y_clean_train, X_removed, y_removed, X_true_clean, y_true_clean, X_val, y_val, 
        label_encoder, num_classes, args, num_iterations=1
    )

    # Print and save results
    print("\nTeacher Model Metrics:")
    for key, value in teacher_metrics.items():
        print(f"{key}: {value}")

    print("\nStudent Model Metrics:")
    for key, value in student_metrics.items():
        print(f"{key}: {value}")

    # Save results
    results_dir = os.path.join(args.result_dir, args.dataset, args.model_type)
    os.makedirs(results_dir, exist_ok=True)

    teacher_results_file = os.path.join(results_dir, f"{model_str}_teacher.csv")
    student_results_file = os.path.join(results_dir, f"{model_str}_student.csv")

    pd.DataFrame([teacher_metrics]).to_csv(teacher_results_file, index=False)
    pd.DataFrame([student_metrics]).to_csv(student_results_file, index=False)

    print(f"\nTeacher results saved to {teacher_results_file}")
    print(f"Student results saved to {student_results_file}")

    # Compare results
    print("\nComparison of performances:")
    print("Teacher Metrics:")
    for key, value in teacher_metrics.items():
        print(f"{key}: {value}")
    print("\nStudent Metrics:")
    for key, value in student_metrics.items():
        print(f"{key}: {value}")

    # Save comparison results
    comparison_file = os.path.join(results_dir, f"{model_str}_teacher_student_comparison.csv")
    pd.DataFrame({'Teacher': teacher_metrics, 'Student': student_metrics}).to_csv(comparison_file, index=False)
    print(f"\nComparison results saved to {comparison_file}")

if __name__ == '__main__':
    main()