import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import wandb


#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        self.args = args
        if self.args.start_from_epoch > 0:
            self.model.load_state_dict(os.path.join("transformer_checkpoints", f"transformer_epoch_{self.args.start_from_epoch}.pt"))
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        step = self.args.start_from_epoch * len(train_loader)
        with tqdm(train_loader) as pbar:
            for image in pbar:
                image = image.to(self.args.device)
                pbar.set_description(f"Training Epoch {self.args.epochs}, learning rate: {self.optim.param_groups[0]['lr']}")
                logits, target  = self.model(image)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                step += 1
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        return train_loss / len(train_loader)

                

    def eval_one_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0
        with tqdm(val_loader) as pbar:
            for image in pbar:
                image = image.to(self.args.device)
                pbar.set_description(f"Evaluating Epoch {self.args.epochs}")
                logits, target  = self.model(image)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def configure_optimizers(self,args):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        if args.lr_schedule == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5], gamma=0.1)
        return optimizer,scheduler 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='/home/pp037/DLP/Lab_5/EXPERIMENT/checkpoints/transformer_last.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=3, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--save_root', type=str, default='./checkpoints/', help='Save CKPT root path')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Which epoch to start from.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate.')

    parser.add_argument('--wandb-run-name', type=str, default='transformer', help='Name of the wandb run.')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    parser.add_argument('--lr_schedule', type=str, default='MultiStepLR', help='Learning rate schedule.')
    args = parser.parse_args()
    os.makedirs(args.save_root, exist_ok=True)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:    
    train_loss_history = []
    val_loss_history = []
    wandb.init(project="Lab5",
            #    mode='disabled',
               config=vars(args),
               name=args.wandb_run_name,
               save_code=True)
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        val_loss = train_transformer.eval_one_epoch(val_loader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), os.path.join(args.save_root, f"transformer_epoch_{epoch}.pt"))
        torch.save(train_transformer.model.transformer.state_dict(), os.path.join(args.save_root, "transformer_last.pt"))
        train_transformer.scheduler.step()
        print(f"Current Epoch {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")
        if args.lr_schedule == 'warmup':
            lr=train_transformer.scheduler.lr
        else:
            lr=train_transformer.scheduler.get_last_lr()[0]
        wandb.log({
            "Train Loss": train_loss,
            "Valid Loss": val_loss,
            "Learning Rate": lr
        })
    with open("loss.csv", "w") as f:
        f.write("train_loss,val_loss\n")
        for train_loss, val_loss in zip(train_loss_history, val_loss_history):
            f.write(f"{train_loss},{val_loss}\n")

