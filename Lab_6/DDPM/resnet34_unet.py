# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=None)
class ResidualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride=1):
        #https://john850512.files.wordpress.com/2019/02/residual.png
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            temp = self.shortcut(x)
            out += temp
        out = F.relu(out)
        return out
    

class ResNet34(nn.Module):
    def __init__(self, channels, classes):
        super(ResNet34, self).__init__()
        self.channels = channels
        self.classes = classes
        self.pre_layer = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False),#kernel是7，然後下一層shape會變成一半，所以padding是3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        )
        self.layer1 = self.resnet_layer(64, 64, 3) #第一層沒有使用shortcut，所以不用設stride
        self.layer2 = self.resnet_layer(64, 128, 4, 2)
        self.layer3 = self.resnet_layer(128, 256, 6, 2)
        self.layer4 = self.resnet_layer(256, 512, 3, 2)

        self.fc = nn.Linear(512, classes)
    
    def resnet_layer(self, in_channels, out_channels, block_num,stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1,block_num):#從1開始，因為第一個block已經加進去了，stride也不用設，因為通常第一個block會有stride
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.pre_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.shape[3]) 
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
    
class ResNet34_UNet(nn.Module):
    def __init__(self, channels, classes):
        super(ResNet34_UNet, self).__init__()
        self.channels = channels
        self.classes = classes
        self.resnet = ResNet34(channels,classes)
        self.resnet_layers = list(self.resnet.children())
        self.conv1 = self.resnet_layers[0]
        self.conv2 = self.resnet_layers[1]
        self.conv3 = self.resnet_layers[2]
        self.conv4 = self.resnet_layers[3]
        self.conv5 = self.resnet_layers[4]
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True))
        self.up_1 = Up(768,768,32)
        self.up_2 = Up(288, 288, 32)
        self.up_3 = Up(160, 160, 32)
        self.up_4 = Up(96, 96, 32)
        self.last_up=nn.ConvTranspose2d(32,32,kernel_size=2,stride=2)
        self.final_conv = nn.Conv2d(32, classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logging.info(f'Input shape: {x.shape}') # [1, 3, 256, 256]
        x1 = self.conv1(x)
        logging.info(f'x1 shape: {x1.shape}') # [1, 64, 64, 64]
        x2 = self.conv2(x1)
        logging.info(f'x2 shape: {x2.shape}') # [1, 64, 64, 64]
        x3 = self.conv3(x2)
        logging.info(f'x3 shape: {x3.shape}') # [1, 128, 32, 32]
        x4 = self.conv4(x3)
        logging.info(f'x4 shape: {x4.shape}') # [1, 256, 16, 16]
        x5 = self.conv5(x4)
        logging.info(f'x5 shape: {x5.shape}') # [1, 512, 8, 8]
        x = self.bottleneck(x5)
        logging.info(f'bottleneck shape: {x.shape}') # [1, 256, 8, 8]
        x = self.up_1(x, x5)
        logging.info(f'up1 shape: {x.shape}') # [1, 32, 16, 16]
        x = self.up_2(x, x4)
        logging.info(f'up2 shape: {x.shape}') # [1, 32, 32, 32]
        x = self.up_3(x, x3)
        logging.info(f'up3 shape: {x.shape}') # [1, 32, 64, 64]
        x = self.up_4(x, x2)
        logging.info(f'up4 shape: {x.shape}') # [1, 32, 128, 128]
        x = self.last_up(x)
        logging.info(f'last_up shape: {x.shape}') # [1, 32, 256, 256]
        x = self.final_conv(x)
        logging.info(f'Output shape: {x.shape}') # [1, 3, 256, 256]
        x = self.softmax(x)
        return x

    def resnet_layer(self, in_channels, out_channels, block_num,stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1,block_num):#從1開始，因為第一個block已經加進去了，stride也不用設，因為通常第一個block會有stride
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

class Up(nn.Module):
    def __init__(self, in_channels,mid_channels, out_channels):
        '''
        2x2 卷積「上卷積」，將特徵通道數量減半，與下採樣路徑中相應裁剪的特徵圖進行串聯，以及兩個3x3卷積，每個卷積後面跟著一個 ReLU。
        '''
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , mid_channels, kernel_size=2, stride=2)
        self.up_conv = Double_3x3_Conv(mid_channels, out_channels)

    def forward(self, x2, x1):
        concat=torch.cat([x1,x2],dim=1)
        x = self.up(concat)
        return self.up_conv(x)

def module_test():
    channels = 3 
    num_classes = 3
    batch_size = 1  
    height, width = 256, 256 

    model = ResNet34_UNet(channels, num_classes)
    dummy_input = torch.randn(batch_size, channels, height, width)
    output = model(dummy_input)
    print("Output shape:", output.shape)

if __name__ == '__main__':
    module_test()

