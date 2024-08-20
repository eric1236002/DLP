import torch
import torch.nn as nn
from diffusers import UNet2DModel
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, labels_num=24, embedding_label_size=8) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(labels_num, embedding_label_size)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + labels_num,
            out_channels=3,
            time_embedding_type="positional",
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 512, 512),  # More channels -> more parameters
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D", 
                "DownBlock2D",
                "AttnDownBlock2D", 
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", 
                "AttnUpBlock2D", 
                "UpBlock2D",
                "AttnUpBlock2D", 
                "UpBlock2D",
            ),
        )
    def forward(self, x, t, label):
        batch_size, channels, width, height = x.shape
        embeded_label = label.view(batch_size, label.shape[1], 1, 1).expand(batch_size, label.shape[1], width, height)
        unet_input = torch.cat((x, embeded_label), 1)
        unet_output = self.model(unet_input, t).sample
        return unet_output 
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    t = torch.randint(0, 1000, (x.shape[0],)).long()
    #one hot label
    label = torch.randint(0, 24, (x.shape[0],)).long()
    label = F.one_hot(label, num_classes=24).float()  # 轉換為浮點型
    unet = Unet()
    unet_output = unet(x, t, label)
    print(unet_output.shape)
