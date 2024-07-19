# implement your training script here
'''
Implement the code for training the SCCNet model, including functions
related to training, losses, optimizer, backpropagation, etc, remember to save
the model weight.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import Dataloader
import model.SCCNet as SCCNet
import utils
import logging
import time
from tqdm import tqdm
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import tester
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# init logger
logging.basicConfig(level=logging.DEBUG)
log_dir = os.path.join(os.getcwd(), 'D:\Cloud\DLP\lab2\logs')
date_time = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
os.makedirs(log_dir, exist_ok=True)


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

def train(epoch,data,finetune_model,numClasses, timeSample, Nu, C, Nc, Nt, dropoutRate,lr,weight_decay,scheduler,batch_size,padding1,padding2):
    training_accuracy = []
    training_loss = []
    validation_accuracy = []
    validation_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)
    
    writer = SummaryWriter(log_dir=log_dir+'/'+data+'/'+date_time)



    # load dataset
    if data == 'FT':
        dataset = Dataloader.MIBCI2aDataset(mode='finetune',data=data)
        model = SCCNet.SCCNet(numClasses, timeSample, Nu, C, Nc, Nt, dropoutRate)
        model.load_state_dict(torch.load(finetune_model))
    else:
        dataset = Dataloader.MIBCI2aDataset(mode='train',data=data)
        model = SCCNet.SCCNet(numClasses, timeSample, Nu, C, Nc, Nt, dropoutRate)
    #(22,438) nchannels =22 ,time = 438
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    # print(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=30, verbose=True,delta=0.005)


    model.train()
    for t in range(epoch):
        correct_train = 0
        total_train = 0
        current_loss = [] 
        for features, labels in train_loader:
            # logging.info('features shape befoe unsqueeze: %s', features.shape)
            features = features.unsqueeze(1).float()
            # logging.info('features shape: %s', features.shape)
            features, labels = features.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted = torch.max(outputs.data, 1)[1]
            total_train += len(labels)
            correct_train += (predicted == labels).float().sum()
            current_loss.append(loss.item())
        loss = np.mean(current_loss)
        accuracy = 100 * correct_train / total_train

        if scheduler == 'ReduceLROnPlateau':
            scheduler.step(loss)

        training_accuracy.append(accuracy)
        training_loss.append(loss)
        
        writer.add_scalar('Loss/train', loss, t)
        writer.add_scalar('Accuracy/train', accuracy, t)
        
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss_list = []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.unsqueeze(1).float()
                features, labels = features.to(device), labels.to(device).long()

                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss_list.append(loss.item())
                predicted = torch.max(outputs.data, 1)[1]
                total_val += len(labels)
                correct_val += (predicted == labels).float().sum()
        
        val_loss = np.mean(val_loss_list)
        val_accuracy = 100 * correct_val / total_val

        validation_accuracy.append(val_accuracy)
        validation_loss.append(val_loss)

        writer.add_scalar('Loss/val', val_loss, t)
        writer.add_scalar('Accuracy/val', val_accuracy, t)
        print(f'Epoch: {t}, Train Accuracy: {accuracy:.3f}, Train Loss: {loss:.3f}, Val Accuracy: {val_accuracy:.3f}, Val Loss: {val_loss:.3f}')


        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # save model
    torch.save(model.state_dict(), 'D:\Cloud\DLP\lab2\weight\\'+data+date_time+'.pth')
    logging.info('Model saved as '+data+date_time+'.pth')
    test_accuracy,test_loss=tester.test(data,'D:\Cloud\DLP\lab2\weight\\'+data+date_time+'.pth',numClasses, timeSample, Nu, C, Nc, Nt, dropoutRate,lr,weight_decay,scheduler,batch_size,padding1,padding2)
    print('Test Accuracy: %.3f, Test Loss: %.3f' % (test_accuracy,test_loss))
    writer.add_hparams(
        {'numClasses': numClasses, 'timeSample': timeSample, 'Nu': Nu, 'C': C, 'Nc': Nc, 'Nt': Nt, 'dropoutRate': dropoutRate, 'lr': lr, 'weight_decay': weight_decay,'scheduler':scheduler,'batch_size':batch_size},
        {'accuracy': test_accuracy, 'loss': test_loss}
        )
    writer.close()
if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--epoch", type=int, default=800)
    argparse.add_argument("--data", type=str, default='LOSO')
    argparse.add_argument("--finetune_model", type=str, default=None)
    argparse.add_argument("--numClasses", type=int, default=4)
    argparse.add_argument("--timeSample", type=int, default=438)
    argparse.add_argument("--Nu", type=int, default=22)
    argparse.add_argument("--C", type=int, default=22)
    argparse.add_argument("--Nc", type=int, default=20)
    argparse.add_argument("--Nt", type=int, default=1)
    argparse.add_argument("--dropoutRate", type=float, default=0.5)
    argparse.add_argument("--lr", type=float, default=0.001)
    argparse.add_argument("--weight_decay", type=float, default=0.0001)
    argparse.add_argument("--scheduler", type=str, default=None)
    argparse.add_argument("--batch_size", type=int, default=512)
    argparse.add_argument("--padding1", default=(0,0))
    argparse.add_argument("--padding2", default=(0,5))
    args = argparse.parse_args()


    train(args.epoch,args.data,args.finetune_model ,args.numClasses, args.timeSample, args.Nu, args.C, args.Nc, args.Nt, args.dropoutRate,args.lr,args.weight_decay,args.scheduler,args.batch_size,args.padding1,args.padding2)