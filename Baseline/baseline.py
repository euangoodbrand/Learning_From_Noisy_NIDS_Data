# -*- coding:utf-8 -*-
from __future__ import print_function 
import os
from matplotlib.colors import LinearSegmentedColormap
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import MLPNet
import argparse, sys
import numpy as np
import datetime
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt # plotting
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from torch.optim import AdamW
import csv
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, ADASYN

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
parser.add_argument('--dataset', type=str, help='cicids', default='cicids')
parser.add_argument('--n_epoch', type=int, default=150)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=500)
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

    for i, (data, labels, _) in enumerate(train_loader):  # Corrected line
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

    class_names = label_encoder.classes_
    cleaned_class_names = [clean_class_name(name) for name in class_names]    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }

     # Class accuracy
    unique_labels = np.unique(all_labels)
    class_accuracy = {f'class_{label_encoder.inverse_transform([label])[0]}_acc': np.mean([all_preds[i] == label for i, lbl in enumerate(all_labels) if lbl == label]) for label in unique_labels}
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
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=cleaned_class_names, yticklabels=cleaned_class_names, annot_kws={"fontsize": 14})
    plt.xticks(rotation=45, ha='right', fontsize=14)  
    plt.yticks(rotation=45, va='top', fontsize=14)
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')  
    plt.ylabel('Actual', fontsize=14, fontweight='bold') 
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    matrix_dir = os.path.join(args.result_dir, 'confusion_matrix')
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir)
    
    matrix_filename = f"{args.dataset}_{args.noise_rate}_{args.data_augmentation.replace(' ', '_').lower()}_confusion_matrix.png"
    plt.savefig(os.path.join(matrix_dir, matrix_filename), bbox_inches='tight')
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

def main():

    preprocessed_file_path = 'final_dataframe.csv'

    if os.path.exists(preprocessed_file_path):
        print("Concatonated dataset already exists")
        # Load the preprocessed DataFrame from the saved CSV file
        df = pd.read_csv(preprocessed_file_path)
    else:
        print("Dataset doesn't exists: generating...")

        df1 = pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        df2=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        df3=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")
        df4=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
        df5=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        df6=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        df7=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv")
        df8=pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv")
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)

        # nRowsRead = 20000  # specify 'None' to read all rows
        # df6 = pd.read_csv("data/cicids2017/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", nrows=nRowsRead)
        # df = pd.concat([df6], ignore_index=True)

        df.reset_index(drop=True, inplace=True)
        nRow, nCol = df.shape
        df.columns = df.columns.str.strip()
        print("Saving concatonated data")

        df.to_csv(preprocessed_file_path, index=False)

    # Convert your DataFrame columns to numpy arrays if not already done
    label_encoder = LabelEncoder()
    labels_np = label_encoder.fit_transform(df['Label'].values)

    features_np = df.drop('Label', axis=1).values.astype(np.float32)


    # Check for inf and -inf values
    print("Contains inf: ", np.isinf(features_np).any())
    print("Contains -inf: ", np.isneginf(features_np).any())

    # Check for NaN values
    print("Contains NaN: ", np.isnan(features_np).any())

    # Replace inf/-inf with NaN
    features_np[np.isinf(features_np) | np.isneginf(features_np)] = np.nan

    # Impute NaN values with the median of each column
    imputer = SimpleImputer(strategy='median')
    features_np_imputed = imputer.fit_transform(features_np)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit on the imputed features data and transform it to standardize
    features_np_standardized = scaler.fit_transform(features_np_imputed)

    # Generate indices for your dataset, which will be used for splitting
    indices = np.arange(len(labels_np))

    # Correctly split the standardized and imputed dataset
    X_train, X_test, y_train, y_test = train_test_split(features_np_standardized, labels_np, test_size=0.3, random_state=42)

    # Data Augmentation Handling based on the user's choice
    if args.data_augmentation == 'smote':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    elif args.data_augmentation == 'undersampling':
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        # important print as undersampling can cause division by zero in extreme data imbalance cases like cic-ids-2017
        print(f"Samples after undersampling: {len(X_train)}") 

    elif args.data_augmentation == 'oversampling':
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
    elif args.data_augmentation == 'adasyn':
        adasyn = ADASYN(random_state=42)
        X_train, y_train = adasyn.fit_resample(X_train, y_train)

    # Introduce label noise after data augmentation
    y_train_noisy, noise_or_not = introduce_label_noise(y_train, noise_rate=args.noise_rate)

    # Prepare datasets for model training and evaluation
    train_dataset = CICIDSDataset(X_train, y_train_noisy, noise_or_not)
    test_dataset = CICIDSDataset(X_test, y_test, np.zeros(len(y_test), dtype=bool))  # Test dataset should have no noise

    # DataLoader setup (as per previous code)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=args.num_workers, drop_last=True, shuffle=False)
    # Define models
    print('building model...')

    torch.manual_seed(234565678)
    clf1 = MLPNet()
    clf1.apply(weights_init)
    clf1.cuda()
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)

    # Define optimizers
    print(clf1.parameters)

    # Check directories exist for saving
    result_dir_path = os.path.dirname(txtfile)
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)

    # Ensure the result directory exists
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)



    # Initialize CSV with dynamic columns after initial evaluation
    initial_metrics = evaluate(test_loader, clf1,label_encoder, args)
    header_fields = list(initial_metrics.keys())


    with open(txtfile, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_fields)
        writer.writeheader()

    # Print and save initial evaluation results
    print('Initial Evaluation - Metrics: ' + ', '.join([f'{k}: {v:.4f}' for k, v in initial_metrics.items() if isinstance(v, float)]))

    # Record the initial metrics
    with open(txtfile, "a", newline='', encoding='utf-8') as csvfile:  
        writer = csv.DictWriter(csvfile, fieldnames=header_fields)
        writer.writerow(initial_metrics)

    # Training loop
    for epoch in range(1, args.n_epoch):
        # Training routine
        clf1.train()
        adjust_learning_rate(optimizer1, epoch)
        criterion = nn.CrossEntropyLoss()

        train_acc = train(train_loader, clf1, optimizer1, criterion, epoch)

        # Re-evaluate after training
        metrics = evaluate(test_loader, clf1, label_encoder, args)
        print(f'Epoch {epoch}/{args.n_epoch} - Test Metrics: ' + ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items() if isinstance(v, float)]))

        # Write metrics to CSV
        with open(txtfile, "a", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header_fields)
            writer.writerow(metrics)

if __name__ == '__main__':
    main()