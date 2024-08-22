import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from timm import create_model
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, init_segmentor

# 使用 thop 计算 FLOP
from thop import profile
from mmseg.datasets import build_dataset
from mmcv import Config

# 加载模型配置和数据集
cfg = Config.fromfile('configs/upernet/upernet_convnext_base_512x512_160k_ade20k.py')
model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg')).cuda()

# 加载语义分割数据集
dataset = build_dataset(cfg.data.train)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# FLOP 计算
input_tensor = torch.randn(1, 3, 512, 512).cuda()
flops, params = profile(model, inputs=(input_tensor,))
print(f'FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.2f} M')

# 训练和评估
def train_model(model, data_loader, optimizer, num_epochs=5):
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader):
            images = data['img'].data[0].cuda()
            targets = data['gt_semantic_seg'].data[0].cuda()
            outputs = model(images)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    end_time = time.time()
    return end_time - start_time

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# 训练模型
train_time = train_model(model, data_loader, optimizer, num_epochs=5)
print(f"Training Time: {train_time:.2f} seconds")

# 推理时间评估
def measure_inference_time(model, input_tensor):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        output = model(return_loss=False, img=[input_tensor])
    end_time = time.time()
    return end_time - start_time

# 测试推理时间
inference_time = measure_inference_time(model, input_tensor)
print(f"Inference Time: {inference_time:.4f} seconds")
