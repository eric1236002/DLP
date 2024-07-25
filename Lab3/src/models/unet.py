# Implement your UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F
class UNet(torch.nn.Module):
    def __init__(self, channels, classes):
        super(UNet, self).__init__()
        self.channels = channels
        self.classes = classes
        self.conv1 = Double_3x3_Conv(channels, 64)
        self.down_1 = Down(64, 128)
        self.down_2 = Down(128, 256)
        self.down_3 = Down(256, 512)
        self.down_4 = Down(512, 1024)
        self.up_1 = Up(1024, 512)
        self.up_2 = Up(512, 256)
        self.up_3 = Up(256, 128)
        self.up_4 = Up(128, 64)
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x1_1 = self.down_1(x1)
        x1_2 = self.down_2(x1_1)
        x1_3 = self.down_3(x1_2)
        x1_4 = self.down_4(x1_3)
        x2_4 = self.up_1(x1_4, x1_3)
        x2_3 = self.up_2(x2_4, x1_2)
        x2_2 = self.up_3(x2_3, x1_1)
        x2_1 = self.up_4(x2_2, x1)
        x = self.final_conv(x2_1)
        x = self.softmax(x)
        return x

class Double_3x3_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        '''
        它由兩個 3x3 卷積(unpadded convolutions)重複應用組成，每個卷積後面跟著一個 (ReLU)
        '''
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        下採樣=Maxpooling+Double_3x3_Conv
        '''
        super().__init__()
        self.Down = nn.Sequential(
            nn.MaxPool2d(2),
            Double_3x3_Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.Down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        2x2 卷積「上卷積」，將特徵通道數量減半，與下採樣路徑中相應裁剪的特徵圖進行串聯，以及兩個3x3卷積，每個卷積後面跟著一個 ReLU。
        '''
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.up_conv = Double_3x3_Conv(in_channels, out_channels)

    def forward(self, x2, x1):
        # 與下採樣路徑中相應裁剪的特徵圖進行串聯
        x2= self.up(x2)  
        #先將x2裁剪成x1的大小 (1024-512)//2=256
        disX = x1.size()[2] - x2.size()[2]
        disY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [disX // 2,disX//2,disY // 2,disY//2]) 
        concat=torch.cat([x1,x2],dim=1)
        return self.up_conv(concat)

