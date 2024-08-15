import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
import argparse
import dataloader, unet, resnet34_unet
import os
from tqdm import tqdm
from eval.evaluator import evaluation_model
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class CDDPM():
    def __init__(self, net, optimizer, scheduler, args, loss_fn, noise_scheduler):
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataload = DataLoader(dataloader.iclevr(path=args.dataset_path, mode=args.mode, partial=args.partial), 
                                   batch_size=args.batch_size, shuffle=True)
        self.args = args
        self.loss_fn = loss_fn
        self.noise_scheduler = noise_scheduler
        self.evaluator = evaluation_model(path=args.dataset_path)
        self.num_classes = 24
        self.save_path = args.save_path

        # Ensure save directory exists
        os.makedirs(os.path.join(self.save_path, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'images'), exist_ok=True)

    def save_model(self,epoch):
        
        # 創建一個包含所有要保存的狀態的字典
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        
        # 定義保存路徑
        save_path = os.path.join(self.save_path, 'model', f'{self.args.model}_checkpoint_{epoch}.pth')
        
        # 保存整個字典
        torch.save(checkpoint, save_path)
        
        print(f"model saved at {save_path}")

    def load_model(self):
        if self.args.load_ckpt:
            try:
                checkpoint = torch.load(self.args.load_ckpt, map_location=device)
                self.net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                print(f"load model from {self.args.load_ckpt} start training from epoch {self.args.start_epoch}")

            except Exception as e:
                print(f"load checkpoint error: {e}")
                print("start training from epoch 0")
        else:
            print("no checkpoint file, start training from epoch 0")

    def train(self):
        loss_save = []
        for epoch in range(self.args.start_epoch, self.args.epochs):
            loss_temp = 0
            for x, y in tqdm(self.dataload):
                x = x.to(device)
                y = y.to(device)
                noise = torch.randn(x.shape).to(device)
                timestamp = torch.randint(0, self.args.timesteps, (x.shape[0],), device=device).long()
                noise_x = self.noise_scheduler.add_noise(x, noise, timestamp)
                perd_noise = self.net(noise_x, timestamp, y)


                loss = self.loss_fn(perd_noise, x)
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_temp += loss.item()

            if (epoch + 1) % self.args.save_per == 0:
                self.save_model(epoch)
            if (epoch + 1) % self.args.test_per == 0:
                test_acc = self.eval('test')
                new_test_acc = self.eval('new_test')
                print(f'Epoch {epoch+1}/{self.args.epochs}, Accuracy: {test_acc}, {new_test_acc}')
                wandb.log({'test_acc': test_acc, 'new_test_acc': new_test_acc})
            loss_save.append(loss_temp / len(self.dataload))
            print(f'Epoch {epoch+1}/{self.args.epochs}, Loss: {loss_save[-1]}')
            wandb.log({'loss': loss_save[-1]})

    @torch.no_grad()
    def eval(self, mode='test'):
        if mode == 'test':
            dataload = DataLoader(dataloader.iclevr(path=self.args.dataset_path, mode='test'), 
                                  batch_size=self.args.batch_size, shuffle=False)
        elif mode == 'new_test':
            dataload = DataLoader(dataloader.iclevr(path=self.args.dataset_path, mode='new_test'), 
                                  batch_size=self.args.batch_size, shuffle=False)
        
        acc_temp = 0
        num_images = 0
        for label in dataload:
            label = label.to(device)
            images = torch.randn(self.args.batch_size, 3, 64, 64).to(device)
            
            # 保存去噪過程
            denoising_process = [images.cpu()]
            
            for j, t in tqdm(enumerate(self.noise_scheduler.timesteps)):
                pred = self.net(images, t, label)
                images = self.noise_scheduler.step(pred, t, images).prev_sample
                
                if j % (len(self.noise_scheduler.timesteps) // 8) == 0:
                    denoising_process.append(images.cpu())
            
            # 使用 make_grid 生成網格
            image_grid = vutils.make_grid(images.cpu(), nrow=8, normalize=True, padding=2)
            denoising_grid = vutils.make_grid(torch.cat(denoising_process), nrow=8, normalize=True, padding=2)
            
            # 保存圖像
            vutils.save_image(image_grid, os.path.join(self.args.save_path, 'images', f'{mode}_{num_images}.png'))
            vutils.save_image(denoising_grid, os.path.join(self.args.save_path, 'images', f'{mode}_denoising_{num_images}.png'))
            
            acc = self.evaluator.eval(images, label)
            acc_temp += acc
            num_images += 1

        print(f'Accuracy: {acc_temp/len(dataload)}')
        return acc_temp / len(dataload)

    def generated_images(self):
        images = torch.randn(self.args.batch_size, 3, 64, 64).to(device)
        return images


def main(args):
    if args.model == 'unet':
        net = unet.Unet()
    elif args.model == 'resnet34_unet':
        net = resnet34_unet.ResNet34Unet()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = nn.MSELoss()
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps, beta_schedule="squaredcos_cap_v2")

    cddpm = CDDPM(net, optimizer, scheduler, args, loss_fn, noise_scheduler)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
 
    if args.mode == 'train':
        if args.start_epoch > 0:
            cddpm.load_model()
        cddpm.train()
    elif args.mode == 'test':
        cddpm.load_model()
        test_acc = cddpm.eval('test')
        new_test_acc = cddpm.eval('new_test')
    print(f'Test Acc: {test_acc}, New Test Acc: {new_test_acc}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='unet or resnet34_unet')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--dataset_path', type=str, default='/home/pp037/DLP/Lab_6/DDPM/eval/', help='dataset path')
    parser.add_argument('--partial', type=float, default=1.0, help='partial') 
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--save_path', type=str, default='/home/pp037/DLP/Lab_6/DDPM/', help='save checkpoint path')
    parser.add_argument('--load_ckpt', type=str, default='/home/pp037/DLP/Lab_6/DDPM/model/unet_checkpoint_3.pth', help='load checkpoint path')
    parser.add_argument('--save_per', type=int, default=2, help='save model every n epochs')
    parser.add_argument('--test_per', type=int, default=20, help='test model every n epochs')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--timesteps', type=int, default=1000, help='timesteps')
    parser.add_argument('--wandb_run_name', type=str, default='unet', help='wandb run name')

    args = parser.parse_args()
    wandb.init(project="DDPM",
        #    mode='disabled',
            config=vars(args),
            name=args.wandb_run_name,
            save_code=True)
    main(args)
