# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
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
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False),#kernel是7，然後下一層shape會變成一半，所以padding是3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        )
        self.conv2 = self.resnet_layer(64, 64, 3) 
        self.conv3 = self.resnet_layer(64, 128, 4, 2)
        self.conv4 = self.resnet_layer(128, 256, 6, 2)
        self.conv5 = self.resnet_layer(256, 512, 3, 2)
        self.neck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size= 3, padding= 1, bias= False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size= 3, padding= 1, bias= False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.up1 = Up(1024+512, 512)
        self.up2 = Up(512+256, 256)
        self.up3 = Up(256+128, 128)
        self.up4 = Up(128+64, 64)
        self.up5 = Up(64, 32)

        self.final_conv = nn.Conv2d(32, classes, kernel_size=1)
        # self.softmax = nn.Softmax(dim=1)

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
        x = self.neck(x5)
        logging.info(f'x shape: {x.shape}') # [1, 1024, 4, 4]
        x = self.up1(x, x5)
        logging.info(f'x shape: {x.shape}') # [1, 512, 8, 8]
        x = self.up2(x, x4)
        logging.info(f'x shape: {x.shape}') # [1, 256, 16, 16]
        x = self.up3(x, x3)
        logging.info(f'x shape: {x.shape}') # [1, 64, 64, 64]
        x = self.up4(x, x2)
        logging.info(f'x shape: {x.shape}') # [1, 64, 128, 128]
        x = self.up5(x)
        logging.info(f'x shape: {x.shape}') # [1, 32, 256, 256]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        logging.info(f'x shape: {x.shape}') # [1, 32, 256, 256]
        x = self.final_conv(x)
        logging.info(f'Output shape: {x.shape}') # [1, 3, 256, 256]
        return x

    def resnet_layer(self, in_channels, out_channels, block_num,stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1,block_num):#從1開始，因為第一個block已經加進去了，stride也不用設，因為通常第一個block會有stride
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = Double_3x3_Conv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1=F.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=True)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)
def module_test():
    channels = 3 
    num_classes = 3
    batch_size = 1  
    height, width = 256, 256 

    model = ResNet34_UNet(channels, num_classes)
    dummy_input = torch.randn(batch_size, channels, height, width)
    output = model(dummy_input)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    module_test()