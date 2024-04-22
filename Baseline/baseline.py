# -*- coding:utf-8 -*-
from __future__ import print_function 
from model import MLPNet

# General Imports
import os
import re
import csv
import datetime
import argparse, sys

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
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
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


def introduce_label_noise(labels, noise_rate=args.noise_rate):
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

model_str = args.model_type + '_' + args.dataset + '_%s_' % args.data_augmentation + '_' + str(args.noise_rate)

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


def train(train_loader, model, optimizer, criterion, epoch):
    print('Training...')
    model.train()  # Set model to training mode
    train_total = 0
    train_correct = 0

    for i, (data, labels, _) in enumerate(train_loader):
        data, labels = data.cuda(), labels.cuda()
        # Forward pass: Compute predicted outputs by passing inputs to the model
        logits = model(data)

        # Calculate the batch's accuracy
        _, predicted = torch.max(logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Calculate the loss
        loss = criterion(logits, labels)

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()
        if (i + 1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  % (epoch + 1, args.n_epoch, i + 1, len(train_loader), 100. * train_correct / train_total, loss.item()))

    train_acc = 100. * train_correct / train_total
    return train_acc

def clean_class_name(class_name):
    # Replace non-standard characters with spaces
    cleaned_name = re.sub(r'[^a-zA-Z0-9]+', ' ', class_name)
    # Remove "web attack" in any case combination
    cleaned_name = re.sub(r'\bweb attack\b', '', cleaned_name, flags=re.IGNORECASE)
    return cleaned_name.strip()  # Strip leading/trailing spaces

def evaluate(test_loader, model, label_encoder, args):
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
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0)
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

    # Define colors
    colors = ["#FFFFFF", "#B9F5F1", "#C8A8E2"]  
    # Create a color map
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)    
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(12, 10))

    # Format the title according to the rules
    model_type_formatted = args.model_type.capitalize()
    dataset_formatted = args.dataset.capitalize()
    data_augmentation_display = "No Data Augmentation" if args.data_augmentation is None or args.data_augmentation.lower() == 'none' else args.data_augmentation.capitalize()

    title = f"{model_type_formatted} on {dataset_formatted} with {data_augmentation_display}, Noise Rate: {args.noise_rate}"
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=cleaned_class_names, yticklabels=cleaned_class_names, annot_kws={"fontsize": 14}, linewidths=0.75, linecolor='black')    
    plt.xticks(rotation=45, ha='right', fontsize=14)  
    plt.yticks(rotation=45, va='top', fontsize=14)
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')  
    plt.ylabel('Actual', fontsize=14, fontweight='bold') 
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Adding border around the color bar
    cbar = plt.gca().collections[0].colorbar
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor("black")

    matrix_dir = os.path.join(args.result_dir, 'confusion_matrix')
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)
    
    matrix_filename = f"{args.dataset}_{args.noise_rate}_{args.data_augmentation.replace(' ', '_').lower()}_confusion_matrix.png"
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
        npz_file_path = 'data/Windows_PE/real_world/malware.npz'
        with np.load(npz_file_path) as data:
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    elif args.dataset == 'BODMAS':
        npz_file_path = 'data/Windows_PE/synthetic/malware.npz'
        with np.load(npz_file_path) as data:
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

    features_np, labels_np = apply_data_augmentation(X_train, y_train, args.data_augmentation)
    labels_np, noise_or_not = introduce_label_noise(labels_np, noise_rate=args.noise_rate)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    results = []
    fold = 0

    # Check directories exist for saving
    result_dir_path = os.path.dirname(txtfile)
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)

    # Ensure the result directory exists
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Prepare CSV file
    os.makedirs(args.result_dir, exist_ok=True)

    if args.dataset == 'BODMAS':
        with open(txtfile, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro'] + \
                        [f'Class {label+1}_acc' for label in label_encoder.classes_]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    elif args.dataset == 'CIC_IDS_2017':
        with open(txtfile, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro'] + \
                        [f'{label}_acc' for label in label_encoder.classes_]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    elif args.dataset == 'windows_pe_real':
        labels = ["Benign", "VirLock", "WannaCry", "Upatre", "Cerber",
                "Urelas", "WinActivator", "Pykspa", "Ramnit", "Gamarue",
                "InstallMonster", "Locky"]
        with open(txtfile, "w", newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Fold', 'Epoch', 'accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_micro', 'f1_macro'] + \
                        [f'{label}_acc' for label in labels]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            

    for train_idx, val_idx in skf.split(features_np, labels_np):
        fold += 1
        X_train_fold, X_val_fold = features_np[train_idx], features_np[val_idx]
        y_train_fold, y_val_fold = labels_np[train_idx], labels_np[val_idx]
        noise_or_not_train, noise_or_not_val = noise_or_not[train_idx], noise_or_not[val_idx]

        train_dataset = CICIDSDataset(X_train_fold, y_train_fold, noise_or_not_train)
        val_dataset = CICIDSDataset(X_val_fold, y_val_fold, noise_or_not_val)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        model = MLPNet(num_features=X_train_fold.shape[1], num_classes=len(np.unique(y_train_fold)), dataset=args.dataset).cuda()
        model.apply(weights_init)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = CrossEntropyLoss()

        for epoch in range(1, args.n_epoch + 1):
            train(train_loader, model, optimizer, criterion, epoch)
            metrics = evaluate(val_loader, model, label_encoder, args)

            # Logging to CSV
            metrics.update({'Fold': fold, 'Epoch': epoch})
            with open(txtfile, "a", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(metrics)

        results.append(metrics)

    print("Training completed. Results from all folds:")
    for i, result in enumerate(results, 1):
        print(f'Results Fold {i}:', result)

if __name__ == '__main__':
    main()