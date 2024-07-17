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
# init logger
logging.basicConfig(level=logging.DEBUG)

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

def train(epoch,data):
    training_accuracy = []
    training_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)

    # load dataset
    train_dataset = Dataloader.MIBCI2aDataset(mode='train',data=data)
    # train_dataset = Dataloader.MIBCI2aDataset(mode='finetune',data=data)
    #(22,438) nchannels =22 ,time = 438
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # load model
    model = SCCNet.SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=1, dropoutRate=0.5)
    # model.load_state_dict(torch.load('D:\Cloud\DLP\lab2\weight\LOSO_53.pth'))
    model.to(device)
    # print(model)
    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    #use different loss function
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    early_stopping = EarlyStopping(patience=20, verbose=True)

    # start training
    model.train()
    for t in range(epoch):
        correct_train = 0
        total_train = 0
        current_loss = [] 
        for i, (features, labels) in enumerate(train_loader):
            # to float
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
        training_accuracy.append(accuracy)

        print('Epoch: %d,Accuracy: %.3f, Loss: %.3f' % (t,accuracy, loss))
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # save model
    data += time.strftime('%d-%H-%M-%S', time.localtime(time.time()))
    torch.save(model.state_dict(), 'D:\Cloud\DLP\lab2\weight\\'+data+'.pth')
    logging.info('Model saved as '+data+'.pth')

if __name__ == '__main__':

    data=['LOSO', 'SD', 'FT']
    train(500,data[0])