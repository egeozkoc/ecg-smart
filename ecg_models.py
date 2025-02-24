import torch
import torch.nn as nn
import torchvision.models as models

# ECG-SMART-NET Architecture ###########################################################
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
        self.fc = nn.Linear(512, num_classes)
        self.dropout = dropout
        self.do = nn.Dropout(p=0.2)

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
        out = self.relu(out)
        
        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.dropout:
            out = self.do(out)

        return out  
########################################################################################

# Temporal ResNet-18 Architecture ######################################################
class Temporal(nn.Module):
    def __init__(self, num_classes=2):
        super(Temporal, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,7), stride=(1,2), padding=(0,3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []

        layers.append(ResidualBlock2D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
########################################################################################

# Pretrained ResNet-18 Architecture ####################################################
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights='DEFAULT')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
########################################################################################