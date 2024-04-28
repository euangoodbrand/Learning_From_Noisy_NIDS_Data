# -*- coding:utf-8 -*-
from __future__ import print_function 
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

from sklearn.neighbors import NearestNeighbors

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

# Co Teaching Specific Imports 
from loss import loss_coteaching, loss_coteaching_plus


for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None 

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='Type of noise to introduce', choices=['uniform', 'class', 'feature','MIMICRY'], default='uniform')
parser.add_argument('--num_gradual', type=int, default=10, help='how many epochs for linear drop rate. This parameter is equal to Ek for lambda(E) in the paper.')
parser.add_argument('--dataset', type=str, help='cicids', choices=['CIC_IDS_2017','windows_pe_real','BODMAS'])
parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=1, help='how many subprocesses to use for data loading')
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus]', default='baseline')
parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
parser.add_argument('--data_augmentation', type=str, choices=['none', 'smote', 'undersampling', 'oversampling', 'adasyn'], default=None, help='Data augmentation technique, if any')
parser.add_argument('--imbalance_ratio', type=float, default=0.0, help='Ratio to imbalance the dataset')
parser.add_argument('--weight_resampling', type=str, choices=['Naive', 'Focal', 'Class-Balance'], default=None, help='Select the weight resampling method if needed')

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
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)  # Calculate the number of samples to corrupt based on the noise rate
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)  # Randomly select indices to corrupt

    new_labels = labels.copy()  # Copy the original labels
    noise_or_not = np.zeros(n_samples, dtype=bool)  # Initialize a mask to track noisy samples

    for idx in noisy_indices:
        original_class = labels[idx]
        # Choose a new label based on the probability distribution from the matrix for the original class
        new_labels[idx] = np.random.choice(np.arange(len(class_noise_matrix[original_class])), p=class_noise_matrix[original_class])
        noise_or_not[idx] = new_labels[idx] != labels[idx]  # Update the noise tracking mask

    return new_labels, noise_or_not

def introduce_mimicry_noise(labels, class_noise_matrix, noise_rate):
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)  # Calculate the number of samples to corrupt based on the noise rate
    noisy_indices = np.random.choice(n_samples, size=n_noisy, replace=False)  # Randomly select indices to corrupt

    new_labels = labels.copy()  # Copy the original labels
    noise_or_not = np.zeros(n_samples, dtype=bool)  # Initialize a mask to track noisy samples

    for idx in noisy_indices:
        original_class = labels[idx]
        # Choose a new label based on the probability distribution from the matrix for the original class
        new_labels[idx] = np.random.choice(np.arange(len(class_noise_matrix[original_class])), p=class_noise_matrix[original_class])
        noise_or_not[idx] = new_labels[idx] != labels[idx]  # Update the noise tracking mask

    return new_labels, noise_or_not


def calculate_feature_thresholds(features):
    # Calculate thresholds for each feature, assuming features is a 2D array
    thresholds = np.percentile(features, 50, axis=0)  # Median as threshold for each feature
    return thresholds


def introduce_feature_dependent_label_noise(features, labels, noise_rate, n_neighbors=5):
    # Initialize the nearest neighbors finder
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 because the point itself is included
    knn.fit(features)
    
    # Find the k-nearest neighbors (including the point itself)
    distances, indices = knn.kneighbors(features)
    
    # Determine the indices that should be noisy
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_rate)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    # Copy labels to prepare for noise introduction
    new_labels = labels.copy()
    noise_or_not = np.zeros(n_samples, dtype=bool)

    for i in noisy_indices:
        # Get the indices of the neighbors excluding the point itself
        neighbor_indices = indices[i][1:]  # skip the first index because it is the point itself
        neighbor_classes = labels[neighbor_indices]
        
        # Identify neighbors with different class labels
        different_class_neighbors = neighbor_indices[neighbor_classes != labels[i]]
        
        if len(different_class_neighbors) > 0:
            # Randomly choose one of the different class neighbors
            chosen_neighbor = np.random.choice(different_class_neighbors)
            new_label = labels[chosen_neighbor]
            new_labels[i] = new_label
            noise_or_not[i] = True if new_label != labels[i] else False

    return new_labels, noise_or_not


def introduce_uniform_noise(labels, noise_rate=args.noise_rate):
    n_samples = len(labels)
    n_noisy = int(noise_rate * n_samples)
    noisy_indices = np.random.choice(np.arange(n_samples), size=n_noisy, replace=False)

    # Initialize as all False, indicating no sample is noisy initially
    noise_or_not = np.zeros(n_samples, dtype=bool)  

    # Iterate over the randomly selected indices to introduce noise
    unique_labels = np.unique(labels)
    for idx in noisy_indices:
        original_label = labels[idx]
        # Exclude the original label to ensure the new label is indeed different
        possible_labels = np.delete(unique_labels, np.where(unique_labels == original_label))
        # Randomly select a new label from the remaining possible labels
        new_label = np.random.choice(possible_labels)
        labels[idx] = new_label  # Assign the new label
        noise_or_not[idx] = True  # Mark this index as noisy

    return labels, noise_or_not


def apply_imbalance(features, labels, ratio, min_samples_per_class=1, downsample_half=True):
    if ratio == 0:
        print("No imbalance applied as ratio is 0.")
        return features, labels

    if ratio >= 1:
        raise ValueError("Imbalance ratio must be less than 1.")

    # Identify the unique classes and their counts
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    
    # Determine which classes to downsample
    # Example: Downsample the first half of the sorted unique classes
    if downsample_half:
        downsample_classes = unique[:n_classes // 2]
    else:
        downsample_classes = unique

    indices_to_keep = []
    for cls in unique:
        class_indices = np.where(labels == cls)[0]
        if cls in downsample_classes:
            # Downsample these classes
            n_minority = np.min(counts)  # Use the smallest class count as the base for downsampling
            n_majority_new = int(n_minority * ratio)
            if len(class_indices) > n_majority_new:
                keep_indices = np.random.choice(class_indices, max(n_majority_new, min_samples_per_class), replace=False)
            else:
                keep_indices = class_indices  # Keep all samples if class count is below the target
        else:
            # Keep all samples from the classes not being downsampled
            keep_indices = class_indices
        
        indices_to_keep.extend(keep_indices)

    indices_to_keep = np.array(indices_to_keep)
    np.random.shuffle(indices_to_keep)  # Shuffle indices to mix classes
    
    return features[indices_to_keep], labels[indices_to_keep]



def compute_weights(labels, no_of_classes, beta=0.9999, gamma=2.0):
    # Count each class's occurrence
    samples_per_class = np.bincount(labels.cpu().numpy(), minlength=no_of_classes)

    if args.weight_resampling == 'Naive':
        # Naive re-weighting: weights are inversely proportional to the class frequencies
        weights = 1.0 / samples_per_class
    elif args.weight_resampling == 'Class-Balance':
        # Class-Balance re-weighting using the effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
    elif args.weight_resampling == 'Focal':
        # Focal re-weighting: Adjust weights based on class frequency and the focusing parameter gamma
        initial_weights = 1.0 / samples_per_class  # Start with inverse frequency
        focal_weights = initial_weights ** gamma  # Apply focal adjustment
        weights = focal_weights
    else:
        raise ValueError("Unsupported weight computation method")

    # Normalize the weights such that their sum equals the number of classes
    weights = (weights / weights.sum()) * no_of_classes

    # Convert the weights to a PyTorch tensor
    weights = torch.tensor(weights, dtype=torch.float, device='cuda:0')

    # Map weights to the corresponding labels to assign each label its corresponding weight
    weight_per_label = weights[labels]

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

model_str = f"{args.model_type}_{args.dataset}_{'no_augmentation' if args.data_augmentation == 'none' else args.data_augmentation}_{args.noise_type}-noise{args.noise_rate}_imbalance{args.imbalance_ratio}"

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

def train(train_loader, model1, optimizer1, model2, optimizer2, epoch, args, no_of_classes, noise_or_not):
    model1.train()  # Set model1 to training mode
    model2.train()  # Set model2 to training mode

    train_total = 0
    train_correct1 = 0
    train_correct2 = 0
    total_loss1 = 0
    total_loss2 = 0

    for i, (data, labels, indices) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        indices = indices.cpu().numpy().transpose()

        logits1 = model1(data)
        logits2 = model2(data)

        if epoch < init_epoch:
            loss1, loss2, _, _ = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], indices, noise_or_not)
        else:
            if args.model_type == 'coteaching_plus':
                loss1, loss2, _, _ = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], indices, noise_or_not, epoch * i)

        # Apply weights manually if weight resampling is enabled
        if args.weight_resampling is not None:
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

        # Calculate accuracy
        _, predicted1 = torch.max(logits1.data, 1)
        _, predicted2 = torch.max(logits2.data, 1)
        train_total += labels.size(0)
        train_correct1 += (predicted1 == labels).sum().item()
        train_correct2 += (predicted2 == labels).sum().item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()

        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4F, Loss1: %.4f, Loss2: %.4f' 
                  % (epoch + 1, args.n_epoch, i + 1, len(train_loader), 100. * train_correct1 / train_total, 100. * train_correct2 / train_total, total_loss1 / (i + 1), total_loss2 / (i + 1)))

    train_acc1 = 100. * train_correct1 / train_total
    train_acc2 = 100. * train_correct2 / train_total
    return train_acc1, train_acc2


def clean_class_name(class_name):
    # Replace non-standard characters with spaces
    cleaned_name = re.sub(r'[^a-zA-Z0-9]+', ' ', class_name)
    # Remove "web attack" in any case combination
    cleaned_name = re.sub(r'\bweb attack\b', '', cleaned_name, flags=re.IGNORECASE)
    return cleaned_name.strip()  # Strip leading/trailing spaces


def calculate_metrics(all_labels, all_preds, label_encoder, args):
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_average': np.mean(f1_score(all_labels, all_preds, average=None, zero_division=0))  # Average F1 score
    }
    return metrics



def create_confusion_matrix(all_labels, all_preds, label_encoder, args):
    # Define the colormap for the heatmap
    colors = ["#FFFFFF", "#B9F5F1", "#C8A8E2"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    # Determine the class names based on the dataset
    if args.dataset == 'CIC_IDS_2017':
        class_names = [label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))]
    elif args.dataset == 'windows_pe_real':
        class_names = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber", "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue", "InstallMonster", "Locky"]
    elif args.dataset == 'BODMAS':
        class_names = [f"Class {i + 1}" for i in range(10)]
    else:
        class_names = [label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))]

    # Clean the class names
    cleaned_class_names = [clean_class_name(name) for name in class_names]

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, 
                     xticklabels=cleaned_class_names, 
                     yticklabels=cleaned_class_names, 
                     annot_kws={"fontsize": 14})

    # Prepare the title based on various configurations
    resampling_status = 'weight_resampling' if args.weight_resampling is not None else 'no_weight_resampling'
    augmentation = 'No Augmentation' if args.data_augmentation == 'none' else args.data_augmentation.capitalize()
    title = f"{model_str} on {args.dataset.capitalize()} dataset with {augmentation}, {resampling_status.capitalize()}, {args.noise_type}-Noise Rate: {args.noise_rate}, Imbalance Ratio: {args.imbalance_ratio}"
    
    plt.title(title, fontsize=14, fontweight='bold', loc='left', wrap=True)
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=45, va='center', fontsize=14)
    plt.tight_layout()

    # Save the confusion matrix
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

    # Calculate class accuracies for both models
    class_accuracies1 = calculate_class_accuracies(all_labels, all_preds1, label_encoder, args)
    class_accuracies2 = calculate_class_accuracies(all_labels, all_preds2, label_encoder, args)

    metrics1.update(class_accuracies1)
    metrics2.update(class_accuracies2)

    print("Metrics 1",metrics1)
    print("Metrics 2",metrics2)

    # Generate and save confusion matrices for both models if requested
    if save_conf_matrix:
        create_confusion_matrix(all_labels, all_preds1, label_encoder, args)
        create_confusion_matrix(all_labels, all_preds2, label_encoder, args)

    return metrics1, metrics2


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




def apply_data_augmentation(features, labels, augmentation_method):
    if augmentation_method == 'smote':
        return SMOTE(random_state=42).fit_resample(features, labels)
    elif augmentation_method == 'undersampling':
        return RandomUnderSampler(random_state=42).fit_resample(features, labels)
    elif augmentation_method == 'oversampling':
        return RandomOverSampler(random_state=42).fit_resample(features, labels)
    elif augmentation_method == 'adasyn':
        return ADASYN(random_state=42).fit_resample(features, labels)
    return features, labels


def main():
    label_encoder = LabelEncoder()

    # Data loading and preprocessing
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
        X_train, X_test, y_train, y_test = train_test_split(features_np, labels_np, test_size=0.3, random_state=42)

    elif args.dataset == 'windows_pe_real':
        # Load the noisy dataset for training
        npz_noisy_file_path = 'data/Windows_PE/real_world/malware.npz'
        with np.load(npz_noisy_file_path) as data:
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Load the clean dataset for final testing
        npz_clean_file_path = 'data/Windows_PE/real_world/malware_true.npz'
        with np.load(npz_clean_file_path) as data:
            X_clean_test, y_clean_test = data['X_test'], data['y_test']
        y_clean_test = label_encoder.transform(y_clean_test)

    elif args.dataset == 'BODMAS':
        npz_file_path = 'data/Windows_PE/synthetic/malware_true.npz'
        with np.load(npz_file_path) as data:
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    # Apply imbalance to the dataset
    X_train_imbalanced, y_train_imbalanced = apply_imbalance(X_train, y_train, args.imbalance_ratio)

    # Introduce noise to the imbalanced data
    y_train_noisy, noise_or_not = introduce_noise(y_train_imbalanced, X_train_imbalanced, args.noise_type, args.noise_rate)

    # Apply data augmentation to the noisy data
    features_np, labels_np = apply_data_augmentation(X_train_imbalanced, y_train_noisy, args.data_augmentation)

    # Directory for validation and full dataset evaluation results
    results_dir = os.path.join(args.result_dir, args.dataset, args.model_type)
    os.makedirs(results_dir, exist_ok=True)

    # Define the base filename with weight resampling status
    resampling_status = 'weight_resampling' if args.weight_resampling else 'no_weight_resampling'
    if args.weight_resampling:
        base_filename = f"{args.model_type}_{args.dataset}_dataset_{args.data_augmentation if args.data_augmentation != 'none' else 'no_augmentation'}_{args.weight_resampling}_{resampling_status}_{args.noise_type}-noise{args.noise_rate}_imbalance{args.imbalance_ratio}"
    else:
        base_filename = f"{args.model_type}_{args.dataset}_dataset_{args.data_augmentation if args.data_augmentation != 'none' else 'no_augmentation'}_{resampling_status}_{args.noise_type}-noise{args.noise_rate}_imbalance{args.imbalance_ratio}"

    # File paths for CSV and model files
    validation_metrics_file_model1 = os.path.join(results_dir, f"{base_filename}_validation_metrics_model1.csv")
    validation_metrics_file_model2 = os.path.join(results_dir, f"{base_filename}_validation_metrics_model2.csv")
    full_dataset_metrics_file_model1 = os.path.join(results_dir, f"{base_filename}_full_dataset_model1.csv")
    full_dataset_metrics_file_model2 = os.path.join(results_dir, f"{base_filename}_full_dataset_model2.csv")

    final_model_path = os.path.join(results_dir, f"{base_filename}_final_model.pth")



    # Prepare CSV file for validation metrics
    with open(validation_metrics_file_model1, "w", newline='', encoding='utf-8') as csvfile:
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
    with open(validation_metrics_file_model2, "w", newline='', encoding='utf-8') as csvfile:
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

    # Prepare CSV file for test metrics
    with open(full_dataset_metrics_file_model1, "w", newline='', encoding='utf-8') as csvfile:
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

    # Prepare CSV file for test metrics
    with open(full_dataset_metrics_file_model2, "w", newline='', encoding='utf-8') as csvfile:
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


    # Cross-validation training
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    results = []
    fold = 0
    for train_idx, val_idx in skf.split(features_np, labels_np):
        fold += 1
        X_train_fold, X_val_fold = features_np[train_idx], features_np[val_idx]
        y_train_fold, y_val_fold = labels_np[train_idx], labels_np[val_idx]
        noise_or_not_train, noise_or_not_val = noise_or_not[train_idx], noise_or_not[val_idx]

        train_dataset = CICIDSDataset(X_train_fold, y_train_fold, noise_or_not_train)
        val_dataset = CICIDSDataset(X_val_fold, y_val_fold, noise_or_not_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        model1 = MLPNet(num_features=X_train_fold.shape[1], num_classes=len(np.unique(y_train_fold)), dataset=args.dataset).cuda()
        model2 = MLPNet(num_features=X_train_fold.shape[1], num_classes=len(np.unique(y_train_fold)), dataset=args.dataset).cuda()
        model1.apply(weights_init)
        model2.apply(weights_init)
        optimizer1 = optim.Adam(model1.parameters(), lr=args.lr)
        optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.n_epoch):
            no_of_classes = len(np.unique(y_train))  
            train(train_loader, model1, optimizer1, model2, optimizer2, epoch, args, no_of_classes, noise_or_not_train)

            metrics1, metrics2 = evaluate(val_loader, model1, model2, label_encoder, args, save_conf_matrix=False)

            # Write metrics to different CSVs for each model
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

    # Full dataset training
    print("Training on the full dataset...")
    full_train_dataset = CICIDSDataset(features_np, labels_np, noise_or_not)
    full_train_loader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    full_model1 = MLPNet(num_features=features_np.shape[1], num_classes=len(np.unique(labels_np)), dataset=args.dataset).cuda()
    full_model1.apply(weights_init)
    full_optimizer1 = optim.Adam(full_model1.parameters(), lr=args.lr)

    full_model2 = MLPNet(num_features=features_np.shape[1], num_classes=len(np.unique(labels_np)), dataset=args.dataset).cuda()
    full_model2.apply(weights_init)
    full_optimizer2 = optim.Adam(full_model2.parameters(), lr=args.lr)

    full_criterion = nn.CrossEntropyLoss()

    for epoch in range(args.n_epoch):
        no_of_classes = len(np.unique(y_train)) 
        train(full_train_loader, full_model1, full_optimizer1, full_model2, full_optimizer2, epoch, args, no_of_classes, noise_or_not)


    # Evaluate the full dataset and save predictions for both models
    if args.dataset == 'windows_pe_real':
        print("Evaluating on clean dataset...")
        clean_test_dataset = CICIDSDataset(X_clean_test, y_clean_test, np.zeros_like(y_clean_test, dtype=bool))
        clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        full_metrics1, full_metrics2 = evaluate(clean_test_loader, full_model1,full_model2, label_encoder, args, save_conf_matrix=True)
        predictions1, predictions2 = evaluate(clean_test_loader, full_model1,full_model2, label_encoder, args, return_predictions=True)
    else:
        full_metrics1, full_metrics2 = evaluate(full_train_loader, full_model1,full_model2, label_encoder, args, save_conf_matrix=True)
        predictions1, predictions2 = evaluate(full_train_loader, full_model1,full_model2, label_encoder, args, return_predictions=True)

    # Save predictions
    predictions_dir = os.path.join(results_dir, args.dataset, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Save predictions
    predictions_filename1 = os.path.join(predictions_dir, f"{base_filename}_predictions_model1.csv")
    predictions_filename2 = os.path.join(predictions_dir, f"{base_filename}_predictions_model2.csv")
    
    save_predictions(predictions1, predictions_filename1)
    save_predictions(predictions2, predictions_filename2)

    print(full_metrics1)
    # Record the evaluation results for both models without the fold number
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
        writer.writerow(row_data)


if __name__ == '__main__':
    main()