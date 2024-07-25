import argparse
import models.unet
import oxford_pet
import utils
import models.resnet34_unet
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch
import copy
from tqdm import tqdm
import evaluate
import time
import wandb

import wandb.testing

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(args):

    date_time = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(
    # mode='disabled',   
    project="lab3",
    name=date_time,
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.learning_rate,
    "architecture": "UNet",
    "dataset": "Oxford-IIIT Pet Dataset",
    "epochs": args.epochs,
    "batch_size": args.batch_size
    }
)
    print('Using device:', device)

    train_dataset = oxford_pet.load_dataset(args.data_path,mode='train')
    val_dataset = oxford_pet.load_dataset(args.data_path,mode='valid')
    # #測試用縮小資料及
    # train_dataset = torch.utils.data.Subset(train_dataset, np.random.choice(len(train_dataset), 10, replace=False))
    # val_dataset = torch.utils.data.Subset(val_dataset, np.random.choice(len(val_dataset), 10, replace=False))

    if args.model == 'unet':
        model = models.unet.UNet(channels=3,classes=3)
    elif args.model == 'resnet34_unet':
        model = models.resnet34_unet.ResNet34_UNet(channels=3,classes=3)
    else:
        raise ValueError('Model not supported')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    print(f'Training on {len(train_loader)} samples, validating on {len(val_loader)} samples')


    best_val_loss=float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
            
        # 訓練過程
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs} - Training', unit='batch'):
            images = batch['image'].to(device).float()
            masks = batch['mask'].to(device).long().squeeze(1)
                
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)+utils.dice_loss(torch.argmax(outputs, dim=1).unsqueeze(1).float(), masks.float())
            # 計算Dice Score
            outputs = torch.argmax(outputs, dim=1).unsqueeze(1)
            dice = utils.dice_score(outputs.float(), masks.float())
            train_dice += dice.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        train_dice /= len(train_dataset)
        wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_dice
        })
        # 驗證過程
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device).float()
                masks = batch['mask'].to(device).long().squeeze(1)
                    
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                    
                # 計算Dice Score
                outputs = torch.argmax(outputs, dim=1).unsqueeze(1)
                dice = utils.dice_score(outputs.float(), masks.float())
                val_dice += dice.item() * images.size(0)
            
        val_loss /= len(val_dataset)
        val_dice /= len(val_dataset)
        wandb.log({
                'epoch': epoch + 1,
                'val_loss': val_loss,
                'val_dice': val_dice
        })
            
        print(f'Epoch {epoch+1}/{args.epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Training Dice: {train_dice:.4f}, Validation Dice: {val_dice:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './saved_models/'+date_time+'.pth')
            print('Model saved')


        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    test_dice=evaluate.evaluate(model, args.data_path, device)
    wandb.log({
        'test_dice': test_dice
    })
    print(f'Test Dice Score: {test_dice:.4f}')
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data',default='dataset\oxford-iiit-pet')
    parser.add_argument('--epochs', '-e', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model', '-m', type=str, default='resnet34_unet', help='model name')
    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)