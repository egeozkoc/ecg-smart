from ecg_models import *
import numpy as np
from glob import glob
from ecg import ECG
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision.models as models
from scipy import signal
import pandas as pd

def train_epoch(model, device, train_dataloader, criterion, optimizer, scaler):
    train_loss = 0
    model.train()

    ys = []
    y_preds = []

    for (x, y) in train_dataloader:

        # undersample the No ACS class
        indices0 = np.where(y == 0)[0]
        indices1 = np.where(y == 1)[0]
        num_samples = np.min([len(indices1), len(indices0)])

        
        if num_samples > 0:

            indices0 = np.random.choice(indices0, num_samples, replace=False)
            indices = np.concatenate([indices0, indices1], axis=0)
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            x = torch.unsqueeze(x, 1)

            x = x.to(device)
            y = y.to(device)

            with torch.amp.autocast(device.type, enabled=True):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=-1)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            ys.append(y)
            y_preds.append(y_pred)

    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    train_loss /= len(y)

    y_pred = y_pred[:, 1]

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred > 0.5)
    prec = precision_score(y, y_pred > 0.5)
    rec = recall_score(y, y_pred > 0.5)
    spec = recall_score(y, y_pred > 0.5, pos_label=0)
    f1 = f1_score(y, y_pred > 0.5)

    return train_loss, auc, acc, prec, rec, spec, f1, ap

def val_epoch(model, device, val_dataloader, criterion):
    val_loss = 0
    model.eval()

    ys = []
    y_preds = []

    with torch.no_grad():
        for (x,y) in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            x = torch.unsqueeze(x, 1)

            with torch.amp.autocast(device.type, enabled=True):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=-1)

            val_loss += loss.item()
            y_pred = y_pred.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            ys.append(y)
            y_preds.append(y_pred)

    y = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    val_loss /= len(y)

    y_pred = y_pred[:, 1]

    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, y_pred > 0.5)
    prec = precision_score(y, y_pred > 0.5)
    rec = recall_score(y, y_pred > 0.5)
    spec = recall_score(y, y_pred > 0.5, pos_label=0)
    f1 = f1_score(y, y_pred > 0.5)

    return val_loss, auc, acc, prec, rec, spec, f1, ap

def get_data(path, selected_outcome):
    
    train_df = pd.read_csv('train_data_{}.csv'.format(selected_outcome))
    val_df = pd.read_csv('val_data_{}.csv'.format(selected_outcome))
    test_df = pd.read_csv('test_data_{}.csv'.format(selected_outcome))
    train_outcomes = train_df['outcome'].to_numpy()
    val_outcomes = val_df['outcome'].to_numpy()
    test_outcomes = test_df['outcome'].to_numpy()
    train_ids = train_df['id'].to_list()
    val_ids = val_df['id'].to_list()
    test_ids = test_df['id'].to_list()

    # get train data
    train_data = []
    for id in train_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg.waveforms['ecg_median']
        ecg = ecg[:,150:-50]
        # ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        ecg = ecg / max_val[:, None]
        train_data.append(ecg)
    train_data = np.array(train_data)

    val_data = []
    for id in val_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg.waveforms['ecg_median']
        ecg = ecg[:,150:-50]
        # ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        ecg = ecg / max_val[:, None]
        val_data.append(ecg)
    val_data = np.array(val_data)

    test_data = []
    for id in test_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg.waveforms['ecg_median']
        ecg = ecg[:,150:-50]
        # ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        ecg = ecg / max_val[:, None]
        test_data.append(ecg)
    test_data = np.array(test_data)

    return train_data, train_outcomes, val_data, val_outcomes, test_data, test_outcomes

if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)
    path = '../ecgs100\\'
    selected_outcome = 'omi' # 'acs'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ECGSMARTNET().to(device)
    # model = Temporal().to(device)

    x_train, y_train, x_val, y_val, x_test, y_test = get_data(path, selected_outcome)

    # model is pretrained ResNet18 ############################################################################################################
    # model = models.resnet18(weights='DEFAULT')
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    # model = model.to(device)
    ############################################################################################################################################

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    lr = 1e-5
    bs = 128
    num_epochs = 200

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    omi_weight = torch.sum(y_val == 0) / torch.sum(y_val == 1)
    val_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, omi_weight], dtype=torch.float32).to(device))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1)
    # cosine annealing lr
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8)
    
    scaler = torch.amp.GradScaler(device=device, enabled=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    best_val_loss = np.inf
    writer = SummaryWriter(log_dir='runs/ecgsmartnet500_{}'.format(selected_outcome))
    count = 0
    for epoch in range(num_epochs):
        if epoch >= 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3


        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss, train_auc, train_acc, train_prec, train_rec, train_spec, train_f1, train_ap = train_epoch(model, device, train_loader, criterion, optimizer, scaler)
        val_loss, val_auc, val_acc, val_prec, val_rec, val_spec, val_f1, val_ap = val_epoch(model, device, val_loader, val_criterion)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('AUC/Train', train_auc, epoch)
        writer.add_scalar('AP/Train', train_ap, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('AUC/Validation', val_auc, epoch)
        writer.add_scalar('AP/Validation', val_ap, epoch) 

        print('Train Loss: {:.3f}, Train AUC: {:.3f}, Train AP: {:.3f}, Train Acc: {:.3f}, Train Prec: {:.3f}, Train Rec: {:.3f}, Train Spec: {:.3f}, Train F1: {:.3f}'.format(train_loss, train_auc, train_ap, train_acc, train_prec, train_rec, train_spec, train_f1))
        print('Val Loss: {:.3f}, Val AUC: {:.3f}, Val AP: {:.3f}, Val Acc: {:.3f}, Val Prec: {:.3f}, Val Rec: {:.3f}, Val Spec: {:.3f}, Val F1: {:.3f}'.format(val_loss, val_auc, val_ap, val_acc, val_prec, val_rec, val_spec, val_f1))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'models/ecgsmartnet500_{}.pt'.format(selected_outcome))
            print('New best model saved')
            count = 0
        else:
            count +=1
        
        if count == 10:
            break
            
    writer.close()
