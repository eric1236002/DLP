# implement SCCNet model

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# init logger
logging.basicConfig(level=logging.DEBUG)
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=1, dropoutRate=0.5,padding1=(0,0),padding2=(0,5)):

        super(SCCNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt), padding=padding1)
        self.batch_norm1 = nn.BatchNorm2d(Nu)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=Nc, kernel_size=(Nu, 12), padding=padding2)
        self.batch_norm2 = nn.BatchNorm2d(Nc)
        self.square = SquareLayer()
        
        self.dropout = nn.Dropout(p=dropoutRate)
        self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))
        self.fc = nn.Linear(in_features=Nc*((timeSample - 62) // 12 + 1), out_features=numClasses)



    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = x.permute(0, 2, 1, 3)

        x =self.conv2(x)
        x = self.batch_norm2(x)
        x = self.square(x)

        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass