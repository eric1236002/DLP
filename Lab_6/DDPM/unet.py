import torch
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps

class Unet(nn.Module):
    def __init__(self, labels_num=24, embedding_label_size=8) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(labels_num, embedding_label_size)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + labels_num,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256, 256,512),  # More channels -> more parameters
            down_block_types=(
                "DownBlock2D", # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D", # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", # a regular ResNet upsampling block
                "AttnUpBlock2D", # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
    def forward(self, x, t, label):
        batch_size, channels, width, height = x.shape
        embeded_label=self.label_embedding(label)
        embeded_label = label.view(batch_size, label.shape[1], 1, 1).expand(batch_size, embeded_label.shape[1], width, height)
        unet_input = torch.cat((x, embeded_label), 1)
        unet_output = self.model(unet_input, t).sample
        return unet_output 