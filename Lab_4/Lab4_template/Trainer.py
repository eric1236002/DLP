import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr

def Caluate_PSNR(img1, img2):
    psnr=[]
    for i in range(1,630):
        psnr.append(Generate_PSNR(img1[i], img2[i][0]).item())
    avg_psnr = sum(psnr)/(len(psnr) - 1) ##check
    return psnr,avg_psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.current_epoch = current_epoch
        self.beta = 0   
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio

    def update(self):
        # TODO
        self.current_epoch=self.current_epoch+1
        if self.kl_anneal_type == 'None':
            self.beta = 1
        else:
            self.beta = self.frange_cycle_linear(n_iter=self.args.num_epoch, n_cycle=self.kl_anneal_cycle, ratio=self.kl_anneal_ratio)
    
    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule
        if self.kl_anneal_type == 'Cyclical':
            if start+step*(self.current_epoch%period)<=stop:
                return start+step*(self.current_epoch%period)
            else:
                return stop
        elif self.kl_anneal_type == 'Monotonic':
            if start+step*(self.current_epoch%period)<=stop and self.current_epoch<period:
                return start+step*(self.current_epoch%period)
            else:
                return stop
        
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        self.train_loss = []
        self.tfr_history = []
        self.avg_psnr = []
        self.kl_loss = []
        self.recon_loss = []
        self.beta_history = []
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            if self.args.tfr_update!=0:
                if self.current_epoch %self.args.tfr_update==0:
                    self.tfr=1
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            epoch_loss = []
            epoch_recon_loss = []
            epoch_kl_loss = []
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, recon_loss, kl_loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

                epoch_loss.append(loss.detach().cpu())
                epoch_recon_loss.append(recon_loss.detach().cpu())
                epoch_kl_loss.append(kl_loss.detach().cpu())
            self.beta_history.append(beta)
            self.recon_loss.append(sum(epoch_recon_loss) / len(epoch_recon_loss))
            self.kl_loss.append(sum(epoch_kl_loss) / len(epoch_kl_loss))
            self.train_loss.append(sum(epoch_loss) / len(epoch_loss))  # 儲存平均損失
            self.tfr_history.append(self.tfr)  # 記錄當前的 Teacher Forcing Ratio
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            self.save_loss_csv()
        self.plot_and_save_loss()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        all_psnr = []
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr_list, avg_psnr = self.val_one_step(img, label)
            if loss =="nan":
                break
            all_psnr.extend(psnr_list)
            self.avg_psnr.append(avg_psnr)
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            print('\nEpoch: {}, PSNR: {}'.format(self.current_epoch, avg_psnr))
        if self.current_epoch % self.args.per_save == 0:
           self.plot_psnr(all_psnr)  # 繪製 PSNR 圖表
        

    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        img = img.to(self.args.device)
        label = label.to(self.args.device)
        loss=torch.tensor(0.0, device=self.args.device)
        recon_loss_sum = torch.tensor(0.0, device=self.args.device)
        kl_loss_sum = torch.tensor(0.0, device=self.args.device)
        for i in range(1, self.train_vi_len):
            frame_current,label_current,frame_pre = img[:,i], label[:,i], img[:,i-1]
            # Encoder取影片的前i張
            if adapt_TeacherForcing and i==1:
                x_pre = self.frame_transformation(frame_pre)
            else:
                x_pre = self.frame_transformation(gen_img)
            x_current = self.frame_transformation(frame_current)
            p_current = self.label_transformation(label_current)


            # Gaussian predictor
            z, mu, logvar = self.Gaussian_Predictor(x_current, p_current)
            # decode fusion
            decoded = self.Decoder_Fusion(x_pre, p_current, z)
            gen_img = self.Generator(decoded)
            #reconstruction loss
            recon_loss = self.mse_criterion(gen_img, frame_current)
            recon_loss_sum += recon_loss
            #kl divergence loss
            kl_loss = kl_criterion(mu, logvar, img.size(0))
            kl_loss_sum += kl_loss

            beta = self.kl_annealing.get_beta()
            loss += recon_loss + beta * kl_loss
        recon_loss_sum /= (self.train_vi_len-1)
        kl_loss_sum /= (self.train_vi_len-1)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()
        return loss, recon_loss_sum, kl_loss_sum

    def val_one_step(self, img, label):
        # TODO
        img = img.to(self.args.device)
        label = label.to(self.args.device)
        pre_img = img[:, 0]
        image_list = [pre_img]
        loss=torch.tensor(0.0, device=self.args.device)
        for i in range(1,self.val_vi_len):
            pose_current, frame_current=label[:,i], img[:,i]
            # Encoder
            frame_feature = self.frame_transformation(pre_img)
            label_feature = self.label_transformation(pose_current)

            # Gaussian predictor
            z= torch.randn(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).to(self.args.device)

            # decode fusion
            decoded = self.Decoder_Fusion(frame_feature, label_feature, z)
            
            generated = self.Generator(decoded)
            pre_img=generated
            image_list.append(generated)
            #reconstruction loss
            recon_loss = self.mse_criterion(generated, frame_current)

            loss += recon_loss 
        psnr_list, avg_psnr = Caluate_PSNR(img[0], image_list)
        loss /= (self.val_vi_len - 1)
        return loss, psnr_list, avg_psnr
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        if self.current_epoch > self.args.tfr_sde:
            self.tfr = max(0, self.tfr - self.args.tfr_d_step)        
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()


    def plot_psnr(self, psnr_list):
        plt.figure()
        plt.plot(psnr_list, label='PSNR per frame')
        plt.axhline(y=np.mean(psnr_list), color='r', linestyle='--', label=f'Average PSNR:{np.mean(psnr_list):.2f}')
        plt.xlabel('Frame')
        plt.ylabel('PSNR')
        plt.title('PSNR per Frame during Validation')
        plt.legend()
        plt.savefig(os.path.join(self.args.save_root, 'epoch_'+str(self.current_epoch)+'_psnr_per_frame.png'))
        plt.close()

    def plot_and_save_loss(self):
        plt.figure()
        plt.plot(self.train_loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.ylim([min(self.train_loss), np.percentile(self.train_loss, 95)])
        plt.savefig(os.path.join(self.args.save_root, 'training_loss.png'))
        plt.close()

        plt.figure()
        plt.plot(self.tfr_history, label='Teacher Forcing Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Teacher Forcing Ratio')
        plt.title('Teacher Forcing Ratio over Epochs')
        plt.legend()
        plt.ylim([min(self.tfr_history), np.percentile(self.tfr_history, 95)])
        plt.savefig(os.path.join(self.args.save_root, 'teacher_forcing_ratio.png'))
        plt.close()

        plt.figure()
        plt.plot(self.kl_loss, label='KL Loss')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.title('KL Loss over Epochs')
        plt.legend()
        plt.ylim([min(self.kl_loss), np.percentile(self.kl_loss, 95)])
        plt.savefig(os.path.join(self.args.save_root, 'kl_loss.png'))
        plt.close()

        plt.figure()
        plt.plot(self.recon_loss, label='Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Loss over Epochs')
        plt.legend()
        plt.ylim([min(self.recon_loss), np.percentile(self.recon_loss, 95)])
        plt.savefig(os.path.join(self.args.save_root, 'reconstruction_loss.png'))
        plt.close()

        plt.figure()
        plt.plot(self.beta_history, label='Beta')
        plt.xlabel('Epoch')
        plt.ylabel('Beta')
        plt.title('Beta over Epochs')
        plt.legend()
        plt.savefig(os.path.join(self.args.save_root, 'beta.png'))
        plt.close()

        plt.figure()
        plt.plot(self.avg_psnr, label='Average PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('Average PSNR over Epochs')
        plt.legend()
        plt.savefig(os.path.join(self.args.save_root, 'avg_psnr.png'))
        plt.close()

    def save_loss_csv(self):
        with open(os.path.join(self.args.save_root, 'training_loss.csv'), 'w') as f:
            f.write('Epoch,Training Loss,Teacher Forcing Ratio,KL Loss,Reconstruction Loss,Beta,Average PSNR\n')
            for i in range(len(self.train_loss)):
                f.write(f'{i},{self.train_loss[i]},{self.tfr_history[i]},{self.kl_loss[i]},{self.recon_loss[i]},{self.beta_history[i]},{self.avg_psnr[i]}\n')
        print('Save training loss to csv file')


def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    set_seed(args.seed)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    # parser.add_argument('--DR',            type=str,   help="Your Dataset Path",default='/home/pp037/DLP/Lab_4/LAB4_Dataset')
    # parser.add_argument('--save_root',     type=str,   help="The path to save your data",default='/home/pp037/DLP/Lab_4/LAB4_Dataset/checkpoint')
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    parser.add_argument('--tfr_update',    type=int,    default=0,   help="Cycle of teacher forcing ratio update")
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical', choices=['Cyclical', 'Monotonic', 'None'],      help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    parser.add_argument('--seed', type=int, default=42)
    

    args = parser.parse_args()
    main(args)
