import torch
import torch.nn as nn
from diffusers import UNet2DModel
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps

class Unet(nn.Module):
    def __init__(self, labels_num=24, embedding_label_size=4) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(labels_num, embedding_label_size)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + labels_num,
            out_channels=3,
            time_embedding_type="positional",
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",  # a regular ResNet upsampling block
            ),
        )
    def forward(self, x, t, label):
        batch_size, channels, width, height = x.shape
        embeded_label = label.view(label.shape[0], label.shape[1], 1, 1).expand(batch_size, label.shape[1], width, height)
        unet_input = torch.cat((x, embeded_label), 1)
        unet_output = self.model(unet_input, t).sample
        return unet_output 