# -*- coding:utf-8 -*-
from __future__ import print_function

from tqdm import tqdm
from model import MLPNet
from MentorNet import MentorNet_arch

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

from MentorMixLoss import MentorMixLoss

for dirname, _, filenames in os.walk('/data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None 

import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma_p', type=float, default=0.75)
    parser.add_argument('--ema', type=float, default=0.05)
    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
    parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.0)
    parser.add_argument('--alpha', type=float, help='corruption rate, should be less than 1', default=2.0)
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
    parser.add_argument('--model_type', type=str, help='[coteaching, coteaching_plus]', default='mentorMix')
    parser.add_argument('--fr_type', type=str, help='forget rate type', default='type_1')
    parser.add_argument('--data_augmentation', type=str, choices=['none', 'smote', 'undersampling', 'oversampling', 'adasyn'], default='none', help='Data augmentation technique, if any')
    parser.add_argument('--imbalance_ratio', type=float, default=0.0, help='Ratio to imbalance the dataset')
    parser.add_argument('--weight_resampling', type=str, choices=['none','Naive', 'Focal', 'Class-Balance'], default='none', help='Select the weight resampling method if needed')
    parser.add_argument('--feature_add_noise_level', type=float, default=0.0, help='Level of additive noise for features')
    parser.add_argument('--feature_mult_noise_level', type=float, default=0.0, help='Level of multiplicative noise for features')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for L2 regularization. Default is 0 (no regularization).')

    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Set batch size and learning rate based on the dataset
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

    # Mock MentorNet output for testing
    output = torch.tensor([0.5324, 0.5199, 0.5310], requires_grad=True)
    print(f"Output from MentorNet: {output}")

    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Hyper Parameters
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

    if args.forget_rate is None:
        forget_rate=args.noise_rate
    else:
        forget_rate=args.forget_rate

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
        for items in test_loader:
            if len(items) == 5:
                data, labels, _, _, _ = items  # Adjusting unpacking for five returned items
            else:
                data, labels, _ = items  # Original unpacking for three items
            
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

        resampling_status = 'weight_resampling' if args.weight_resampling is not None else 'no_weight_resampling'

        if args.weight_resampling:
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

def train(args, MentorNet, StudentNet, train_dataloader, optimizer_M, optimizer_S, scheduler_M, scheduler_S, BCE_loss, CE_loss, loss_p_prev, epoch):
    MentorNet.train()
    StudentNet.train()
    MentorNet_loss = 0
    StudentNet_loss = 0
    p_bar = tqdm(total=len(train_dataloader), desc='Training')

    for batch_idx, (inputs, targets, v_true, v_label, index) in enumerate(train_dataloader):
        inputs, targets, v_label = inputs.cuda(), targets.cuda(), v_label.cuda()

        # Forward pass for StudentNet
        outputs = StudentNet(inputs)
        loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Sort losses and calculate the threshold loss_p
        sorted_losses, _ = torch.sort(loss)
        loss_p = sorted_losses[int(len(sorted_losses) * args.gamma_p)].item()
        loss_diff = loss - loss_p

        # Train MentorNet
        v_predicted = MentorNet(v_label, args.n_epoch, epoch, loss.detach(), loss_diff.detach())  # Ensure MentorNet does not backprop through StudentNet
        v_true_adjusted = ((loss_diff < 0) * 1).float().cuda()  # Simple simulation for v_true

        loss_M = BCE_loss(v_predicted, v_true_adjusted)
        MentorNet_loss += loss_M.item()

        optimizer_M.zero_grad()
        loss_M.backward()
        optimizer_M.step()

        # Train StudentNet
        loss_S = (loss * v_predicted.detach()).mean()  # Detach v_predicted to prevent gradients flowing into MentorNet
        StudentNet_loss += loss_S.item()

        optimizer_S.zero_grad()
        loss_S.backward()  # No need to retain graph here, as MentorNet does not depend on StudentNet's gradients
        optimizer_S.step()

        p_bar.update(1)
        p_bar.set_postfix({'Student Loss': StudentNet_loss / (batch_idx + 1), 'Mentor Loss': MentorNet_loss / (batch_idx + 1)})

    p_bar.close()
    return loss_p


def train_student(args, MentorNet, StudentNet, train_dataloader, optimizer_S, scheduler_S, loss_p_prev, loss_p_second_prev, epoch):
    StudentNet.train()
    train_loss = 0
    p_bar = tqdm(range(train_dataloader.__len__()))

    loss_average = 0
    for batch_idx, (inputs, targets, _, v_label, index) in enumerate(train_dataloader):
        loss, loss_p_prev, loss_p_second_prev, v = MentorMixLoss(args, MentorNet, StudentNet, inputs, targets, v_label, loss_p_prev, loss_p_second_prev, epoch)
        
        # Apply weights manually if weight resampling is enabled
        # if args.weight_resampling != 'none':
        #     weights = compute_weights(targets, no_of_classes=np.unique(train_dataloader))
        #     loss = (loss * weights).mean() 

        # Update v
        train_dataloader.dataset.update_v_labels(index, v.long())

        optimizer_S.zero_grad()
        loss.backward()
        optimizer_S.step()
        train_loss += loss.item()
        p_bar.set_description("Train Epoch: {epoch}/{epochs:2}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. loss: {loss:.4f}.".format(
                    epoch=epoch + 1,
                    epochs=args.n_epoch,
                    batch=batch_idx + 1,
                    iter=train_dataloader.__len__(),
                    lr=scheduler_S.optimizer.param_groups[0]['lr'],
                    loss=train_loss/(batch_idx+1))
                    )
        p_bar.update()
    p_bar.close()
    return loss_p_prev, loss_p_second_prev


def test(args, StudentNet, test_dataloader, optimizer_S, scheduler_S, epoch):
    StudentNet.eval()
    test_loss = 0
    acc = 0
    p_bar = tqdm(range(test_dataloader.__len__()))
    with torch.no_grad():
        for batch_idx, items in enumerate(test_dataloader):
            if len(items) == 5:
                inputs, targets, _, _, _ = items  # Adjusting unpacking for five returned items
            else:
                inputs, targets = items  # Original unpacking for two items
            
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = StudentNet(inputs)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()
            p_bar.set_description("Test Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.4f}.".format(
                    epoch=1,
                    epochs=1,
                    batch=batch_idx + 1,
                    iter=test_dataloader.__len__(),
                    lr=scheduler_S.optimizer.param_groups[0]['lr'],
                    loss=test_loss/(batch_idx+1)))
            p_bar.update()
            acc += (outputs.argmax(dim=1) == targets).sum().item()
    p_bar.close()
    acc = acc / test_dataloader.dataset.__len__()
    print('Accuracy :' + '%0.4f' % acc)
    return acc


def main():
    print("Main function started")

    # Optimizer and scheduler setup
    learning_rate = 0.1
    weight_decay = 0.0002
    momentum = 0.9
    nesterov = False

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
    print("Original dataset:")
    print(f"Length of X_train: {len(X_train)}")
    print(f"Length of y_train: {len(y_train)}")
    print("Class distribution in original dataset:", {label: np.sum(y_train == label) for label in np.unique(y_train)})

    # Apply imbalance to the training dataset
    X_train_imbalanced, y_train_imbalanced = apply_imbalance(X_train, y_train, args.imbalance_ratio)

    # Print class distribution after applying imbalance
    print("Before introducing noise:")
    print(f"Length of X_train_imbalanced: {len(X_train_imbalanced)}")
    print(f"Length of y_train_imbalanced: {len(y_train_imbalanced)}")
    print("Class distribution after applying imbalance:", {label: np.sum(y_train_imbalanced == label) for label in np.unique(y_train_imbalanced)})

    # Introduce noise to the imbalanced data
    y_train_noisy, noise_or_not = introduce_noise(y_train_imbalanced, X_train_imbalanced, args.noise_type, args.noise_rate)

    # Apply feature noise
    X_train_imbalanced = feature_noise(torch.tensor(X_train_imbalanced), add_noise_level=args.feature_add_noise_level, mult_noise_level=args.feature_mult_noise_level).numpy()

    # Print class distribution after introducing noise
    print("Before augmentation:")
    print(f"Length of X_train_imbalanced: {len(X_train_imbalanced)}")
    print(f"Length of y_train_noisy: {len(y_train_noisy)}")
    print(f"Length of noise_or_not: {len(noise_or_not)}")
    print("Class distribution after introducing noise:", {label: np.sum(y_train_noisy == label) for label in np.unique(y_train_noisy)})

    # Apply data augmentation to the noisy data
    X_train_augmented, y_train_augmented = apply_data_augmentation(X_train_imbalanced, y_train_noisy, args.data_augmentation)

    if args.data_augmentation in ['smote', 'adasyn', 'oversampling']:
        noise_or_not = np.zeros(len(y_train_augmented), dtype=bool) 

    # Print class distribution after data augmentation
    print("After augmentation:")
    print(f"Length of X_train_augmented: {len(X_train_augmented)}")
    print(f"Length of y_train_augmented: {len(y_train_augmented)}")
    print(f"Length of noise_or_not (adjusted if necessary): {len(noise_or_not)}")
    print("Class distribution after data augmentation:", {label: np.sum(y_train_augmented == label) for label in np.unique(y_train_augmented)})

    # Cross-validation training
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    results = []
    fold = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_augmented, y_train_augmented), start=1):      
        if max(train_idx) >= len(noise_or_not) or max(val_idx) >= len(noise_or_not):
            print("IndexError: Index is out of bounds for noise_or_not array.")
            continue

        X_train_fold, X_val_fold = X_train_augmented[train_idx], X_train_augmented[val_idx]
        y_train_fold, y_val_fold = y_train_augmented[train_idx], y_train_augmented[val_idx]
        noise_or_not_train, noise_or_not_val = noise_or_not[train_idx], noise_or_not[val_idx]

        train_dataset = CICIDSDataset(X_train_fold, y_train_fold, noise_or_not_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        val_dataset = CICIDSDataset(X_clean_test, y_clean_test, np.zeros(len(y_clean_test), dtype=bool))
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        # Initialize MentorNet with more complex architecture
        mentornet = MentorNet_arch().cuda()  

        studentnet = MLPNet(num_features=X_train.shape[1], num_classes=len(np.unique(y_train)), dataset=args.dataset).cuda()

        mentornet.apply(weights_init)
        studentnet.apply(weights_init)

        optimizer_mentor = optim.Adam(mentornet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_student = optim.Adam(studentnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion_mentor = nn.CrossEntropyLoss()
        criterion_mentor  = nn.CrossEntropyLoss()

        scheduler_mentor = optim.lr_scheduler.StepLR(optimizer_mentor, step_size=30, gamma=0.1)
        scheduler_student = optim.lr_scheduler.StepLR(optimizer_student, step_size=30, gamma=0.1)

        loss_p_prev = 0  # Initialize previous period loss variable

        for epoch in range(args.n_epoch):
            loss_p_prev = train(args, mentornet, studentnet, train_loader, optimizer_mentor, optimizer_student, scheduler_mentor, scheduler_student, criterion_mentor, criterion_mentor, loss_p_prev, epoch)
            
            # Evaluate the model using the custom evaluate function
            evaluation_metrics = evaluate(val_loader, studentnet, label_encoder, args, save_conf_matrix=True, return_predictions=False)
            print(f"Evaluation Metrics for Epoch {epoch+1}: {evaluation_metrics}")

            scheduler_mentor.step()
            scheduler_student.step()

            path_MentorNet = './checkpoint/MentorNet'
            if not os.path.isdir(path_MentorNet):
                os.makedirs(path_MentorNet)

            MentorNet_filename = f"{path_MentorNet}/MentorNet_Fold_{fold}"
            torch.save(mentornet.state_dict(), MentorNet_filename + f'_Epoch_{epoch+1}.pt')

        # second training run for student after mentor has been trained
        
        path_MentorNet = './checkpoint/MentorNet'
        latest_epoch = args.n_epoch  
        MentorNet_filename = f"{path_MentorNet}/MentorNet_Fold_{fold}_Epoch_{latest_epoch}.pt"

        # Load the trained MentorNet
        mentornet = MentorNet_arch().cuda()
        mentornet.load_state_dict(torch.load(MentorNet_filename))
        mentornet.eval() 

        # Reinitialize StudentNet
        studentnet = MLPNet(num_features=X_train.shape[1], num_classes=len(np.unique(y_train)), dataset=args.dataset).cuda()
        studentnet.apply(weights_init)  # Apply initializations as before

        # Setup optimizer and scheduler for the new StudentNet
        optimizer_student = optim.Adam(studentnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_student = optim.lr_scheduler.StepLR(optimizer_student, step_size=30, gamma=0.1)

        # Define the loss function for StudentNet, assuming it remains the same
        criterion_student = nn.CrossEntropyLoss()


        path_base = './checkpoint'
        # Directory for saving the specific fold's best model
        fold_dir = f'{path_base}/{args.dataset}/{args.model_type}_Fold_{fold}'
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        best_acc=0
        loss_p_prev = 0
        loss_p_second_prev = 0
        for epoch in range(args.n_epoch):
            loss_p_prev, loss_p_second_prev = train_student(args, mentornet, studentnet, train_loader, optimizer_student, scheduler_student, loss_p_prev, loss_p_second_prev, epoch)
            
            evaluation_metrics = evaluate(val_loader, studentnet, label_encoder, args, save_conf_matrix=True, return_predictions=False)
            print(f"Evaluation Metrics for Epoch {epoch+1}: {evaluation_metrics}")
            acc = test(args, studentnet, val_loader, optimizer_student, scheduler_student, epoch)

            scheduler_student.step()
            if best_acc < acc:
                best_acc = acc
                # Define the path for saving the model
                model_path = f"{fold_dir}/Best_StudentNet_Epoch_{epoch+1}.pth"
                torch.save(studentnet.state_dict(), model_path)
                print(f"Saved improved model to {model_path}")

            # Update metrics with Fold and Epoch at the beginning
            row_data = OrderedDict([('Fold', fold), ('Epoch', epoch)] + list(evaluation_metrics.items()))
            with open(validation_metrics_file, "a", newline='',encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data)

    # Initialize and train MentorNet on the full dataset
    print("Initializing and training MentorNet on the full dataset...")
    mentornet = MentorNet_arch().cuda()
    studentnet = MLPNet(num_features=X_train_augmented.shape[1], num_classes=len(np.unique(y_train_augmented)), dataset=args.dataset).cuda()

    mentornet.apply(weights_init)
    studentnet.apply(weights_init)

    optimizer_mentor = optim.Adam(mentornet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_student = optim.Adam(studentnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_mentor = optim.lr_scheduler.StepLR(optimizer_mentor, step_size=30, gamma=0.1)
    scheduler_student = optim.lr_scheduler.StepLR(optimizer_student, step_size=30, gamma=0.1)
    criterion_mentor = nn.CrossEntropyLoss()
    criterion_student = nn.CrossEntropyLoss()

    full_train_dataset = CICIDSDataset(X_train_augmented, y_train_augmented, noise_or_not)
    full_train_loader = DataLoader(dataset=full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    loss_p_prev = 0  # Initialize previous period loss variable for custom train function
    for epoch in range(args.n_epoch):
        loss_p_prev = train(args, mentornet, studentnet, full_train_loader, optimizer_mentor, optimizer_student, scheduler_mentor, scheduler_student, criterion_mentor, criterion_student, loss_p_prev, epoch)
        scheduler_mentor.step()
        scheduler_student.step()

    # Save the trained MentorNet model
    path_MentorNet = './checkpoint/MentorNet'
    if not os.path.isdir(path_MentorNet):
        os.makedirs(path_MentorNet)
    MentorNet_filename = f"{path_MentorNet}/MentorNet_Final_Epoch_{args.n_epoch}.pt"
    torch.save(mentornet.state_dict(), MentorNet_filename)

    # Load the trained MentorNet to guide the training of a new StudentNet
    print("Loading trained MentorNet to guide the training of a new StudentNet...")
    mentornet.load_state_dict(torch.load(MentorNet_filename))
    mentornet.eval()  # Set MentorNet to evaluation mode

    # Reinitialize and train a new StudentNet with guidance from MentorNet
    print("Reinitializing and training a new StudentNet with guidance from MentorNet...")
    new_studentnet = MLPNet(num_features=X_train_augmented.shape[1], num_classes=len(np.unique(y_train_augmented)), dataset=args.dataset).cuda()
    new_studentnet.apply(weights_init)
    optimizer_new_student = optim.Adam(new_studentnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_new_student = optim.lr_scheduler.StepLR(optimizer_new_student, step_size=30, gamma=0.1)
    criterion_new_student = nn.CrossEntropyLoss()

    loss_p_prev = 0
    loss_p_second_prev = 0
    for epoch in range(args.n_epoch):
        loss_p_prev, loss_p_second_prev = train_student(args, mentornet, new_studentnet, full_train_loader, optimizer_new_student, scheduler_new_student, loss_p_prev, loss_p_second_prev, epoch)
        scheduler_new_student.step()

    # Evaluate the final trained new StudentNet
    print("Evaluating the final trained new StudentNet on clean dataset...")
    clean_test_dataset = CICIDSDataset(X_clean_test, y_clean_test, np.zeros_like(y_clean_test, dtype=bool))
    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    full_metrics = evaluate(clean_test_loader, new_studentnet, label_encoder, args, save_conf_matrix=True)
    predictions = evaluate(clean_test_loader, new_studentnet, label_encoder, args, return_predictions=True)

    # Save predictions and results
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
