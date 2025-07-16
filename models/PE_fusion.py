
import random
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(filenames_shuffled, outcomes_shuffled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)


y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

dfDemographics = pd.read_pickle("AllPatientsDemographics.pkl")

class CustomGenerator(data.Dataset):
    def __init__(self, ecg_filenames, labels):
        self.ecg_filenames = ecg_filenames
        self.labels = labels

    def __len__(self):
        return len(self.ecg_filenames)

    def __getitem__(self, idx):
        batch_x_10sec = self.ecg_filenames[idx]
        batch_x_median = batch_x_10sec[:-10] + '_median.npy'
        batch_y = self.labels[idx]
        shortfilenames = os.path.basename(batch_x_10sec)[:-10]
        thesedemographics = dfDemographics.loc[shortfilenames]

        x_median = np.transpose(np.load(batch_x_median), (1, 0))
        x_v1 = np.transpose(np.load(batch_x_10sec), (1, 0))
        age = np.float32(thesedemographics['Age'])/90
        sex = int(thesedemographics['Male'])

        return (x_median, x_v1, age, sex), batch_y


batch_size = 32
train_dataset = CustomGenerator(X_train, y_train)
val_dataset = CustomGenerator(X_val, y_val)
test_dataset = CustomGenerator(X_test, y_test)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size * 3, shuffle=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mp):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.mp = nn.MaxPool2d(kernel_size=(1, mp), stride = (1, mp), padding=(0, int(mp/2-1)))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.mp(x)
        return x

class Mayo(nn.Module):
    def __init__(self, num_classes=2):
        super(Mayo, self).__init__()
        self.convblock1 = ConvBlock(in_channels = 1, out_channels=16, kernel_size=5, mp=2)
        self.convblock2 = ConvBlock(in_channels =16, out_channels=16, kernel_size=5, mp=2)
        self.convblock3 = ConvBlock(in_channels = 16, out_channels=32, kernel_size=5, mp=4)

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 1), padding='valid')
        self.bn1 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class TransformerBranch(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, num_classes):
        super(TransformerBranch, self).__init__()
        self.embedding = nn.Linear(12, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        x = x[:, ::2, :]
        x = self.embedding(x)

        x = self.transformer_block(x)

        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc(x)
        return x

class HybridECGModel(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridECGModel, self).__init__()

        embed_dim = 8
        num_heads = 1
        ff_dim = 8
        dropout_rate = 0.1

        self.cnn_branch = Mayo(num_classes)
        self.att_branch = TransformerBranch(embed_dim, num_heads, ff_dim, dropout_rate, num_classes)

        self.fc1 = nn.Linear(64 + embed_dim, 64) 
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)


        self.fcp = nn.Linear(6, num_classes)

    def forward(self, x):
        X_all_leads, X_V1_lead, age, sex = x
        cnn_output = self.cnn_branch(X_all_leads)
        att_output = self.att_branch(X_V1_lead)
        age = age.unsqueeze(1)
        sex = sex.unsqueeze(1)

        combined_output = torch.cat((cnn_output, att_output, age, sex), dim=1)
        output = self.fcp(combined_output)

        return output

model = HybridECGModel(num_classes=2)


with torch.no_grad():
    for (x_median, x_v1, age, sex), labels in test_loader:
        x_median = x_median.to(device)
        x_v1 = x_v1.to(device)
        age = age.to(device)
        sex = sex.to(device)
        labels = labels.to(device)
        outputs = model((x_median.float(), x_v1.float(), age.float(), sex.float()))
        outputs = softmax(outputs)
        y_pred_proba.append(outputs[:, 1].cpu().numpy())
        y_true.append(labels.cpu().numpy())
