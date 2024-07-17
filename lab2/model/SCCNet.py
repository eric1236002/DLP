# implement SCCNet model

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# init logger
logging.basicConfig(level=logging.DEBUG)
# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, numClasses=0, timeSample=435, Nu=22, C=22, Nc=20, Nt=1, dropoutRate=0.5):
        '''
        The architecture of SCCNet consists of four blocks: 
        the first convolution block, the second convolution block, the pooling block, and the softmax block.
        '''
        super(SCCNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt), stride=(1, 1))
        self.batch_norm1 = nn.BatchNorm2d(Nu)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=Nc, kernel_size=(Nu, 12), stride=(1, 1))
        self.batch_norm2 = nn.BatchNorm2d(Nc)
        self.square = SquareLayer()
        
        self.dropout = nn.Dropout(p=dropoutRate)
        self.pool = nn.AvgPool2d((1, 62), stride=(1, 12))
        self.fc = nn.Linear(in_features=Nc*((timeSample - 62) // 12), out_features=numClasses)



    def forward(self, x):
        # logging.info('input shape: %s', x.shape)
        x = self.conv1(x)
        # logging.info('conv1 shape: %s', x.shape)
        x = self.batch_norm1(x)
        # logging.info('square shape: %s', x.shape)
        x = x.permute(0, 2, 1, 3)
        # logging.info('permute shape: %s', x.shape)
        x =self.conv2(x)
        # logging.info('conv2 shape: %s', x.shape)
        x = self.batch_norm2(x)
        x = self.square(x)

        x = self.dropout(x)
        x = self.pool(x)
        # logging.info('pool shape: %s', x.shape)
        x = x.view(x.size(0), -1)
        # logging.info('view shape: %s', x.shape)
        x = self.fc(x)
        # logging.info('fc shape: %s', x.shape)
        return F.softmax(x, dim=1)

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass