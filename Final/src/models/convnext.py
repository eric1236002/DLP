import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath
from thop import profile
import time
class LayerNorm(nn.Module):
    """
    LayerNorm 支援兩種數據格式: channels_last 或 channels_first。
    channels_last 對應於輸入形狀為 (batch_size, height, width, channels)，
    而 channels_first 對應於 (batch_size, channels, height, width)。
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    """
    ConvNeXt Block 實現。
    DwConv -> 轉換至 (N, H, W, C) -> LayerNorm -> Linear -> GELU -> Linear -> 轉回 (N, C, H, W)
    
    Args:
        dim (int): 輸入通道數
        drop_path (float): 隨機深度率
        layer_scale_init_value (float): Layer Scale 的初始值
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度卷積
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise 卷積，實現為全連接層
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se = SEBlock(dim)  # 添加SEBlock

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # 轉換至 (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # 轉回 (N, C, H, W)
        x = self.se(x)  # 檢用SEBlock
        x = residual + self.drop_path(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(in_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super(ConvNeXt, self).__init__()

        self.num_classes = num_classes
        
        # 下採樣層，包括 Stem 和 3 個中間的卷積層
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stage 設置，每個 Stage 包含多個 Block
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # 上採樣層
        self.upsample_layers = nn.ModuleList()
        for i in range(3, -1, -1):
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[max(i-1, 0)], kernel_size=2, stride=2),
                LayerNorm(dims[max(i-1, 0)], eps=1e-6, data_format="channels_first")
            )
            self.upsample_layers.append(upsample_layer)

        # 額外的上採樣層，用於恢復到原始尺寸
        self.final_upsample = nn.ConvTranspose2d(dims[0], dims[0], kernel_size=2, stride=2)

        # 最後的輸出層
        self.output_layer = nn.Conv2d(dims[0], num_classes, kernel_size=1)

        # 為每個stage創建對應的SEBlock
        self.se_blocks = nn.ModuleList([SEBlock(dim) for dim in dims])

        self.apply(self._init_weights)

    def forward(self, x):
        # 下採樣和特徵提取
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)

        # 上採樣和特徵融合
        for i in range(4):
            x = self.upsample_layers[i](x)
            if i < 3:  # 只在前三次上採樣時進行特徵融合
                # 使用對應維度的SEBlock
                temp = self.se_blocks[2-i](features[2-i])
                x = x + temp

        # 額外的上採樣，恢復到原始尺寸
        x = self.final_upsample(x)

        # 最後的輸出層
        x = self.output_layer(x)

        # 使用 softmax 確保每個像素位置的各個通道和為 1
        x = F.softmax(x, dim=1)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)


def model_test():
    channels = 3 
    num_classes = 3
    batch_size = 1  
    height, width = 256, 256 
    model = ConvNeXt(in_chans=channels, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1, layer_scale_init_value=1e-6, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # 清理 GPU 內存
    torch.cuda.empty_cache()
    
    # 計算輸出形狀
    output = model(dummy_input)
    print("輸出形狀:", output.shape)
    
    # 計算參數數量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數數量: {total_params:,}")
    
    # 計算 FLOPs
    flops, _ = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops:,}")

    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            output_convnext = model(dummy_input)
    convnext_time = time.time() - start_time
    print(f"ConvNeXt 平均推理時間: {convnext_time / 100:.6f} 秒")

if __name__ == "__main__":
    model_test()
