# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.feature_selection import SelectFromModel


from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,BatchNormalization, LeakyReLU, Input
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestClassifier


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score, f1_score

import matplotlib.pyplot as plt

from numpy.random import seed
seed(42)
from tensorflow import random
random.set_seed(42)

from joblib import dump, load

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.special import softmax
from collections import Counter
from sklearn.utils import resample


import random
import warnings

# MODELS

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel=3):
        super(ResidualBlock2D,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel), stride=(1,stride), padding=(0, kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,kernel), stride=(1,1), padding=(0, kernel//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = out1 + self.shortcut(x)
        out1 = self.relu(out1)

        return out1

# ECG-SMART-NET Model
class ECGSMARTNET(nn.Module):
    def __init__(self, num_classes=2, kernel=7, kernel1=3, num_leads=12, dropout=False):
        super(ECGSMARTNET, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,kernel), stride=(1,2), padding=(0,kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.layer1 = self.make_layer(64, 2, stride=1, kernel=kernel1)
        self.layer2 = self.make_layer(128, 2, stride=2, kernel=kernel1)
        self.layer3 = self.make_layer(256, 2, stride=2, kernel=kernel1)
        self.layer4 = self.make_layer(512, 2, stride=2, kernel=kernel1)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(num_leads,1), stride=(1,1), padding=(0,0), bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
        self.dropout = dropout
        self.do = nn.Dropout(p=0.5)

    def make_layer(self, out_channels, num_blocks, stride, kernel):
        layers = []

        layers.append(ResidualBlock2D(self.in_channels, out_channels, stride, kernel))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(self.in_channels, out_channels, 1, kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.do(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.conv2(out)
        out = self.do(out)
        out = self.bn2(out)

        out = self.relu(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        if self.dropout:
            out = self.do(out)

        return out

# MLP Model
class MLPModel(torch.nn.Module):
  def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

  def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        #x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        #x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        #x = self.dropout(x)
        x = self.fc4(x)
        return x


rf_estimators = 50
ecgsmartnet = ECGSMARTNET()
mlp = MLPModel(input_size=(2)*2, num_classes=2)
#input_size = 2*2 because probabilities (p0 and p1) are fed into MLP (could be changed if we only want predictions as inputs to MLP)

# RF Model
rf = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', n_jobs=-1, random_state=42,
                                                            max_features='log2',
                                                            n_estimators=rf_estimators,
                                                            min_samples_split=0.001,
                                                            min_samples_leaf=0.001,
                                                            min_impurity_decrease=0.0,
                                                            bootstrap=True,
                                                            ccp_alpha=0.001,
                                                            max_samples=0.75,
                                                            oob_score=True)

# MODEL TRAINING
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training parameters
num_epochs = 200
batch_size = 128
update_rf_every = 5
learning_rate = 1e-5
patience = 10
best_val_loss = float('inf')
patience_counter = 0
alpha = 2/3

ecgsmartnet = ecgsmartnet.to(device)
mlp = mlp.to(device)
params = list(ecgsmartnet.parameters()) + list(mlp.parameters()) #combined params

optimizer_ecg = torch.optim.Adam(ecgsmartnet.parameters(), lr=learning_rate, weight_decay=0.1)
optimizer_mlp = torch.optim.SGD(mlp.parameters(), lr=1e-3, weight_decay=0.1)
criterion = torch.nn.CrossEntropyLoss()

# X, y
X_train = "Median beats (training)"
X_val = "Median beats (validation)"
X_test = "Median beats (test)"
X_features_train = "Top 148 features (training)"
X_features_val = "Top 148 features (validation)"
X_features_test = "Top 148 features (test)"
y_train = "Outcomes with classes 1 and 2 combined to class 1 (training)"
y_val = "... (validation)"
y_test = "... (test)"
ID_train = X_features_train.index

# Fit RF
rf.fit(X_features_train, y_train) #here I am fitting RF first, and only using predictions in the hybrid model training phase

def create_balanced_batch(X, y, batch_size):
    class_0_indices = torch.where(y == 0)[0]
    class_1_indices = torch.where(y == 1)[0]
    class_2_indices = torch.where(y == 2)[0]

    samples_per_class = batch_size // 2
    samples_per_class = min(samples_per_class, len(class_0_indices), len(class_1_indices))

    class_0_sample = class_0_indices[torch.randint(len(class_0_indices), (samples_per_class,))]
    class_1_sample = class_1_indices[torch.randint(len(class_1_indices), (samples_per_class,))]

    balanced_indices = torch.cat([class_0_sample, class_1_sample])

    return balanced_indices[:batch_size]


# Training
for epoch in range(num_epochs):
    ecgsmartnet.train()
    mlp.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = create_balanced_batch(X_train, y_train, batch_size) #create balanced batches
        IDs = ID_train[indices]
        batch_X, batch_y = X_train[indices], y_train[indices]
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        X_features_batch = X_features_train.loc[IDs] #create batches for features

        outputs_ecg = ecgsmartnet(batch_X) #predictions from ECG-SMART-NET
        outputs_rf = rf.predict_proba(X_features_batch) #predictions from RF
        outputs_ecg = outputs_ecg.detach().cpu().numpy()

        inputs_mlp = np.concatenate((outputs_ecg, outputs_rf), axis=1)
        inputs_mlp = torch.tensor(inputs_mlp, dtype=torch.float32).to(device)
        outputs = mlp(inputs_mlp) #concatenated predictions into MLP

        batch_y = batch_y.squeeze()
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer_ecg.zero_grad()
        optimizer_mlp.zero_grad()
        loss.backward()

        optimizer_ecg.step()
        optimizer_mlp.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


    # Validation
    ecgsmartnet.eval()
    mlp.eval()
    val_loss = 0
    with torch.no_grad():
        X_val, y_val = X_val.to(device), y_val.to(device)
        val_outputs_ecg = ecgsmartnet(X_val)
        val_outputs_rf = rf.predict_proba(X_features_val)
        val_outputs_ecg = val_outputs_ecg.detach().cpu().numpy()
        inputs_mlp = np.concatenate((val_outputs_ecg, val_outputs_rf), axis=1)
        inputs_mlp = torch.tensor(inputs_mlp, dtype=torch.float32).to(device)
        val_outputs = mlp(inputs_mlp)
        y_val_squeezed = y_val.squeeze()
        loss = criterion(val_outputs, y_val_squeezed)
        y_val_np = y_val.cpu().numpy()
        val_outputs = val_outputs.cpu().numpy()
        val_outputs = softmax(val_outputs, axis=1)
        val_loss += loss.item()

        auc_val = roc_auc_score(y_val_np, val_outputs[:,1])
        ap_val = average_precision_score(y_val_np, val_outputs[:,1])

        val_metric = auc_val + (1 - alpha) * ap_val #compute validation metric

        #Test
        X_test, y_test = X_test.to(device), y_test.to(device)
        test_outputs_ecg = ecgsmartnet(X_test)
        test_outputs_rf = rf.predict_proba(X_features_test)
        test_outputs_ecg = test_outputs_ecg.detach().cpu().numpy()
        inputs_mlp =  np.concatenate((test_outputs_ecg, test_outputs_rf), axis=1)
        inputs_mlp = torch.tensor(inputs_mlp, dtype=torch.float32).to(device)
        test_outputs = mlp(inputs_mlp)
        y_test_np = y_test.cpu().numpy()
        test_outputs = test_outputs.cpu().numpy()
        test_outputs = softmax(test_outputs, axis=1)

        auc_test = roc_auc_score(y_test_np, test_outputs[:,1])
        ap_test = average_precision_score(y_test_np, test_outputs[:,1])


    # EarlyStopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        #Save the best model
        torch.save(ecgsmartnet.state_dict(), "best_ecgsmartnet_model_alternate.pth")
        torch.save(mlp.state_dict(), "best_mlp_model_alternate.pth")

    else:
        patience_counter += 1
        print(f"No drop in val loss for {patience_counter} epochs.")

    if patience_counter >= patience:
        print("EarlyStopping triggered.")
        break

    #Log metrics
    print(f"Validation AUC: {auc_val:.4f}, AP: {ap_val:.4f}, Metric: {val_metric:.4f}, Loss: {val_loss:.4f}")
    print(f"Test AUC: {auc_test:.4f}, AP: {ap_test:.4f}")

#Load best models saved before EarlyStopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ecgsmartnet = ECGSMARTNET()
mlp = MLPModel(input_size=(2)*2, num_classes=2)
ecgsmartnet.load_state_dict(torch.load("best_ecgsmartnet_model_alternate.pth"))
mlp.load_state_dict(torch.load("best_mlp_model_alternate.pth"))

ecgsmartnet = ecgsmartnet.to(device)
mlp = mlp.to(device)

#Validation metrics
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
ecgsmartnet.eval()
mlp.eval()
with torch.no_grad():
  X_val, y_val = X_val.to(device), y_val.to(device)
  val_outputs_ecg = ecgsmartnet(X_val)
  val_outputs_rf = rf.predict_proba(X_features_val)
  val_outputs_ecg = val_outputs_ecg.detach().cpu().numpy()
  inputs_mlp =  np.concatenate((val_outputs_ecg, val_outputs_rf), axis=1)
  inputs_mlp = torch.tensor(inputs_mlp, dtype=torch.float32).to(device)
  val_outputs = mlp(inputs_mlp)
  val_predictions = torch.argmax(val_outputs, dim=1)
  val_accuracy = (val_predictions == y_val.clone().detach().long()).float().mean().item()
  y_val_np = y_val.cpu().numpy()
  val_outputs = val_outputs.cpu().numpy()
  val_outputs = softmax(val_outputs, axis=1)
  auc_val = roc_auc_score(y_val_np, val_outputs[:,1])
  ap_val = average_precision_score(y_val_np, val_outputs[:,1])

  #Other metrics
  f1s = []
  thresholds = np.arange(0.01, 1, 0.01)
  for thres in thresholds: #find best threshold maximizing F1 score
    y_pred = (val_outputs[:,1] >= thres).astype(int)
    f1 = f1_score(y_val_np, y_pred)
    f1s.append(f1)
  max_thres = thresholds[np.argmax(f1s)]
  y_pred = (val_outputs[:,1] >= max_thres).astype(int)
  f1 = max(f1s)
  acc = accuracy_score(y_val_np, y_pred)
  tn, fp, fn, tp = confusion_matrix(y_val_np, y_pred).ravel()
  sen = tp / (tp + fn)
  spe = tn / (tn + fp)
  ppv = tp / (tp + fp)
  npv = tn / (tn + fn)
  print(f"Validation Acc: {acc}")
  print(f"Validation AUC: {auc_val}, Validation AP: {ap_val}")
  print(f"Thres: {max_thres}, Sen: {sen}, Spe: {spe}, PPV: {ppv}, NPV: {npv}, F1: {f1}")

#Test metrics
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
ecgsmartnet.eval()
mlp.eval()
with torch.no_grad():
  #Test
  X_test, y_test = X_test.to(device), y_test.to(device)
  test_outputs_ecg = ecgsmartnet(X_test)
  test_outputs_rf = rf.predict_proba(X_features_test)
  test_outputs_ecg = test_outputs_ecg.detach().cpu().numpy()
  inputs_mlp =  np.concatenate((test_outputs_ecg, test_outputs_rf), axis=1)
  inputs_mlp = torch.tensor(inputs_mlp, dtype=torch.float32).to(device)
  test_outputs = mlp(inputs_mlp)
  test_predictions = torch.argmax(test_outputs, dim=1)
  test_accuracy = (test_predictions == y_test.clone().detach().long()).float().mean().item()
  y_test_np = y_test.cpu().numpy()
  test_outputs = test_outputs.cpu().numpy()
  test_outputs = softmax(test_outputs, axis=1)
  auc_test = roc_auc_score(y_test_np, test_outputs[:,1])
  ap_test = average_precision_score(y_test_np, test_outputs[:,1])

  #Other metrics
  f1s = []
  thresholds = np.arange(0.01, 1, 0.01)
  for thres in thresholds: #find best threshold maximizing F1 score
    y_pred = (test_outputs[:,1] >= thres).astype(int) 
    f1 = f1_score(y_test_np, y_pred)
    f1s.append(f1)
  max_thres = thresholds[np.argmax(f1s)]
  y_pred = (test_outputs[:,1] >= max_thres).astype(int)
  f1 = max(f1s)
  acc = accuracy_score(y_test_np, y_pred)
  tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred).ravel()
  sen = tp / (tp + fn)
  spe = tn / (tn + fp)
  ppv = tp / (tp + fp)
  npv = tn / (tn + fn)
  print(f"Validation Acc: {acc}")
  print(f"Validation AUC: {auc_test}, Validation AP: {ap_test}")
  print(f"Thres: {max_thres}, Sen: {sen}, Spe: {spe}, PPV: {ppv}, NPV: {npv}, F1: {f1}")