# -*- coding:utf-8 -*-
from __future__ import print_function
import copy 
from model import MLPNet

# General Imports
import os
import re
import csv
import datetime
import argparse, sys
from collections import OrderedDict

# Maths and Data processing imports
import numpy as np
import pandas as pd

# Plotting Imports
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Sklearn Import
from sklearn.model_selection import StratifiedKFold
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
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader

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
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type=str, help='cicids', choices=['CIC_IDS_2017','windows_pe_real','BODMAS'])
parser.add_argument('--n_epoch', type=int, default=150)
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
parser.add_argument('--epoch_start', default=25, help='epoch start to introduce l_r', type=int)
parser.add_argument('--epoch_update', default=30, help='epoch start to update labels', type=int)
parser.add_argument('--epoch_interval', default=40, help="interval for updating A", type=int)
parser.add_argument('--every_n_epoch', default=10, help='rolling window for estimating f(x)_t', type=int)
parser.add_argument('--two_stage', default=False, help='train NN again with clean data using original cross entropy', type=bool)

parser.add_argument('--feature_add_noise_level', type=float, default=0.0, help='Level of additive noise for features')
parser.add_argument('--feature_mult_noise_level', type=float, default=0.0, help='Level of multiplicative noise for features')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for L2 regularization. Default is 0 (no regularization).')


args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

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


class CICIDSDataset(Dataset):
    def __init__(self, features, labels, noise_flags, device='cuda'):
        self.features = torch.tensor(features, dtype=torch.float32, device=device)
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)
        self.noise_flags = torch.tensor(noise_flags, dtype=torch.bool, device=device)

    def update_labels(self, new_labels, indices):
        # Ensure both new_labels and indices are on the same device as labels
        new_labels = new_labels.to(self.labels.device)
        indices = indices.to(self.labels.device)
        self.labels.index_copy_(0, indices, new_labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], index

    def __len__(self):
        return len(self.labels)



if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

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
    if total_weight == 0:
        weights = np.ones(no_of_classes, dtype=np.float32)
    else:
        weights = (weights / total_weight) * no_of_classes

    # Convert numpy weights to torch tensor and move to the specified device
    weight_per_label = torch.from_numpy(weights).float().to(device)

    # Index weights by labels
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
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) 
       
# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    if fr_type=='type_1':
        rate_schedule = np.ones(args.n_epoch)*forget_rate
        rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
    return rate_schedule

rate_schedule = gen_forget_rate(args.fr_type)
  
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

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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


def train(train_loader, model, optimizer, criterion, epoch, pred_softlabels, indices_updated, args):
    model.train()  # Set model to training mode
    total_loss = 0

    for i, (inputs, labels, indices) in enumerate(train_loader):
        delta = 5 + 0.02 * max(epoch - args.epoch_update + 1, 0)

        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        # Apply weights manually if weight resampling is enabled
        if args.weight_resampling != 'none':
            weights = compute_weights(labels, no_of_classes=outputs.size(1))
            loss = (loss * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update softmax probabilities periodically for LRT
        if epoch >= (args.epoch_update - args.every_n_epoch):
            softmax_outputs = torch.softmax(outputs, dim=1).detach()
            pred_softlabels[indices.cpu(), epoch % args.every_n_epoch, :] = softmax_outputs.cpu().numpy()  # Ensure alignment of indices

        if epoch >= args.epoch_update:
            # Aggregate and average the predictions over the specified rolling window
            pred_softlabels_bar = np.mean(pred_softlabels, axis=1)  
            pred_softlabels_tensor = torch.tensor(pred_softlabels_bar, device=inputs.device)

            # Use only the current batch indices to slice
            batch_pred_softlabels = pred_softlabels_tensor[indices]
            batch_y_tilde = labels.clone()

            # LRT flip scheme should correctly use batch indices
            corrected_labels, _ = lrt_flip_scheme(batch_pred_softlabels, batch_y_tilde, delta)

            # Update labels within the dataset
            train_loader.dataset.update_labels(corrected_labels, indices)

    print(f'Epoch [{epoch + 1}/{args.n_epoch}], Loss: {total_loss / len(train_loader)}')



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
            
    cleaned_class_names = [clean_class_name(name) for name in index_to_class_name.values()]
    

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
    if args.dataset == 'CIC_IDS_2017':
        unique_labels = np.unique(all_labels)
        class_accuracy = {f'{label_encoder.inverse_transform([label])[0]}_acc': np.mean([all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label]) for label in unique_labels}
        metrics.update(class_accuracy)
    elif args.dataset == 'windows_pe_real':
        unique_labels = np.unique(all_labels)
        class_accuracy = {
            f'{index_to_class_name[label]}_acc': np.mean([
                all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label
            ]) for label in unique_labels
        }
        metrics.update(class_accuracy)
    elif args.dataset == 'BODMAS':
        unique_labels = np.unique(all_labels)
        class_accuracy = {
            f'{index_to_class_name[label]}_acc': np.mean([
                all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label
            ]) for label in unique_labels
        }
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


def lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta):
    '''
    The LRT correction scheme.
    pred_softlabels_bar is the prediction of the network which is compared with noisy label y_tilde.
    If the LR is smaller than the given threshold delta, we reject LRT and flip y_tilde to the prediction of pred_softlabels_bar

    Input
    pred_softlabels_bar: rolling average of output after soft layers for past 10 epochs. Could use other rolling windows.
    y_tilde: noisy labels at current epoch
    delta: LRT threshholding

    Output
    y_tilde: new noisy labels after cleaning
    clean_softlabels: soft version of y_tilde
    '''
    ntrain = pred_softlabels_bar.shape[0]
    num_class = pred_softlabels_bar.shape[1]

    for i in range(ntrain):
        cond_1 = not pred_softlabels_bar[i].argmax() == y_tilde[i]
        cond_2 = (pred_softlabels_bar[i].max() / pred_softlabels_bar[i][y_tilde[i]]) > delta
        if cond_1 and cond_2:
            y_tilde[i] = pred_softlabels_bar[i].argmax()

    eps = 1e-2
    clean_softlabels = torch.ones(ntrain, num_class, device=pred_softlabels_bar.device) * eps / (num_class - 1)
    clean_softlabels.scatter_(1, y_tilde.view(-1, 1), 1 - eps)  # Use y_tilde directly without converting to numpy

    return y_tilde, clean_softlabels



def updateA(s, h, rho=0.9):
    '''
    Used to calculate retroactive loss

    Input
    s : output after softlayer of NN for a specific former epoch
    h : logrithm of output after softlayer of NN at current epoch

    Output
    result : retroactive loss L_retro
    A : correction matrix
    '''
    eps = 1e-4
    h = torch.tensor(h, dtype=torch.float32).reshape(-1, 1)
    s = torch.tensor(s, dtype=torch.float32).reshape(-1, 1)
    A = torch.ones(len(s), len(s))*eps
    A[s.argmax(0)] = rho - eps/(len(s)-1)
    result = -((A.matmul(s)).t()).matmul(h)

    return result, A


def map_indices(global_indices, local_indices):
    """
    Map global indices to the indices of the current subset.
    :param global_indices: Array of indices referring to the full dataset.
    :param local_indices: Array of indices representing the current batch or subset.
    :return: Mapped indices valid within the local_indices context.
    """
    # Create a mapping from the full dataset indices to current subset indices
    local_idx_map = {idx: i for i, idx in enumerate(local_indices)}
    # Filter and map global indices to the current subset indices
    mapped_indices = [local_idx_map[idx] for idx in global_indices if idx in local_idx_map]
    return mapped_indices

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

        # Splitting the data into training, test, and a clean test set
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
        
        # Splitting the data into training and a clean test set
        X_train, X_clean_test, y_train, y_clean_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

    # Directory for validation and full dataset evaluation results
    results_dir = os.path.join(args.result_dir, args.dataset, args.model_type)
    os.makedirs(results_dir, exist_ok=True)

    # File paths for CSV and model files
    validation_metrics_file = os.path.join(results_dir, f"{model_str}_validation.csv")
    full_dataset_metrics_file = os.path.join(results_dir, f"{model_str}_full_dataset.csv")
    final_model_path = os.path.join(results_dir, f"{model_str}_final_model.pth")

    # Prepare CSV file for validation metrics
    with open(validation_metrics_file, "w", newline='', encoding='utf-8') as csvfile:
        if args.dataset == 'BODMAS':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro','f1_average'] + \
                        [f'Class {label+1}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'CIC_IDS_2017':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro','f1_average'] + \
                        [f'{label}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'windows_pe_real':
            labels = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                    "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                    "InstallMonster", "Locky"]
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro','f1_average'] + \
                        [f'{label}_acc' for label in labels]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Prepare CSV file for validation metrics
    with open(full_dataset_metrics_file, "w", newline='', encoding='utf-8') as csvfile:
        if args.dataset == 'BODMAS':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro','f1_average'] + \
                        [f'Class {label+1}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'CIC_IDS_2017':
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro','f1_average'] + \
                        [f'{label}_acc' for label in label_encoder.classes_]
        elif args.dataset == 'windows_pe_real':
            labels = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                    "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                    "InstallMonster", "Locky"]
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro','f1_average'] + \
                        [f'{label}_acc' for label in labels]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Print the original dataset sizes and class distribution

    # Apply imbalance to the training dataset
    X_train_imbalanced, y_train_imbalanced = apply_imbalance(X_train, y_train, args.imbalance_ratio)

    # Introduce noise to the imbalanced data
    y_train_noisy, noise_or_not = introduce_noise(y_train_imbalanced, X_train_imbalanced, args.noise_type, args.noise_rate)

    # Apply feature noise
    X_train_imbalanced = feature_noise(torch.tensor(X_train_imbalanced), add_noise_level=args.feature_add_noise_level, mult_noise_level=args.feature_mult_noise_level).numpy()

    # Apply data augmentation to the noisy data
    X_train_augmented, y_train_augmented = apply_data_augmentation(X_train_imbalanced, y_train_noisy, args.data_augmentation)

    if args.data_augmentation in ['smote', 'adasyn', 'oversampling']:
        # Recalculate noise_or_not to match the augmented data size
        noise_or_not = np.zeros(len(y_train_augmented), dtype=bool)  # Adjust the noise_or_not array size and values as needed

    # ------------------------- Initialize Setting ------------------------------
    num_class = len(np.unique(y_train_augmented))

    # Cross-validation training
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    results = []
    fold = 0
    for train_idx, val_idx in skf.split(X_train_augmented, y_train_augmented):
        X_train_fold, X_val_fold = X_train_augmented[train_idx], X_train_augmented[val_idx]
        y_train_fold, y_val_fold = y_train_augmented[train_idx], y_train_augmented[val_idx]
        noise_or_not_train, noise_or_not_val = noise_or_not[train_idx], noise_or_not[val_idx]

        # Initializing datasets and dataloaders
        train_dataset = CICIDSDataset(X_train_fold, y_train_fold, noise_or_not_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        val_dataset = CICIDSDataset(X_clean_test, y_clean_test, np.zeros(len(y_clean_test), dtype=bool))
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        model = MLPNet(num_features=X_train_fold.shape[1], num_classes=len(np.unique(y_train_fold)), dataset=args.dataset).cuda()
        model.apply(weights_init)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        pred_softlabels = np.zeros([len(train_loader.dataset), args.every_n_epoch, num_class], dtype=np.float64)

        # Map global indices to local indices
        global_to_local_index = {idx: i for i, idx in enumerate(train_idx)}

        for epoch in range(args.n_epoch):
            local_indices_updated = [global_to_local_index[idx] for idx in train_idx if idx in global_to_local_index]
            train(train_loader, model, optimizer, criterion, epoch, pred_softlabels, local_indices_updated, args)
            metrics = evaluate(val_loader, model, label_encoder, args)

            # Log results
            print(f"Fold {fold}, Epoch {epoch}, Validation Metrics: {metrics}")

            # Update metrics with Fold and Epoch at the beginning
            row_data = OrderedDict([('Fold', fold), ('Epoch', epoch)] + list(metrics.items()))
            with open(validation_metrics_file, "a", newline='',encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data)

        fold += 1  # Increment the fold counter

    print("Training completed. Results from all folds:")
    for i, result in enumerate(results, 1):
        print(f'Results Fold {i}:', result)

    # Full dataset training with LRT
    print("Training on the full dataset with LRT...")
    # Prepare the full augmented dataset for training
    full_train_dataset = CICIDSDataset(X_train_augmented, y_train_augmented, noise_or_not)
    full_train_loader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    # Prepare the full model
    full_model = MLPNet(num_features=X_train_augmented.shape[1], num_classes=len(np.unique(y_train_augmented)), dataset=args.dataset).cuda()
    full_model.apply(weights_init)
    full_optimizer = optim.Adam(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    full_criterion = CrossEntropyLoss()

    # Initialize place for storing prediction softmaxes for LRT
    pred_softlabels = np.zeros([len(full_train_loader.dataset), args.every_n_epoch, num_class], dtype=np.float64)

    # Train on the full augmented dataset
    for epoch in range(args.n_epoch):
        indices_updated = range(len(y_train_augmented))  # All indices are potentially updated in full training
        train(full_train_loader, full_model, full_optimizer, full_criterion, epoch, pred_softlabels, indices_updated, args)

    print("Training with LRT completed.")

    # Prepare clean data for evaluation
    clean_test_dataset = CICIDSDataset(X_clean_test, y_clean_test, np.zeros_like(y_clean_test, dtype=bool))
    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # Evaluate the full model on the clean dataset
    print("Evaluating on clean dataset...")
    full_metrics = evaluate(clean_test_loader, full_model, label_encoder, args, save_conf_matrix=True)
    predictions = evaluate(clean_test_loader, full_model, label_encoder, args, return_predictions=True)

    # Save predictions
    predictions_dir = os.path.join(args.result_dir, args.dataset, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_filename = os.path.join(predictions_dir, f"{args.dataset}_final_predictions.csv")
    with open(predictions_filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Predicted Label'])
        for pred in predictions:
            writer.writerow([pred])

    print(f"Predictions saved to {predictions_filename}")

    # Record the evaluation results
    row_data = OrderedDict([('Fold', 'Full Dataset'), ('Epoch', epoch)] + list(full_metrics.items()))
    with open(full_dataset_metrics_file, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row_data)

    print("Final evaluation completed.")

if __name__ == '__main__':
    main()
