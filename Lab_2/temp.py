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
from sklearn.model_selection import KFold
import numpy as np
# init logger
logging.basicConfig(level=logging.DEBUG)

def train(epoch,data):
    training_accuracy = []
    training_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)

    # load dataset
    train_dataset = Dataloader.MIBCI2aDataset(mode='train',data=data)
    #(22,438) nchannels =22 ,time = 438
    # 記錄每次的準確率
    results = []
    
    k_folds = 3
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
        # 準備訓練和驗證數據加載器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        train_loader = torch.utils.data.DataLoader(
                          train_dataset, 
                          batch_size=256, 
                          sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
                          train_dataset,
                          batch_size=256, 
                          sampler=test_subsampler)
    
        # 初始化模型
        model = SCCNet.SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=1, dropoutRate=0.5,Nf=1)
        model.to(device)
    
        # 定義優化器和損失函數
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()
    
        # 訓練模型
        model.train()
        for t in range(epoch):
            for i, (features, labels) in enumerate(train_loader):
                features = features.unsqueeze(1).float()
                features, labels = features.to(device), labels.to(device).long()
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
        # 驗證模型
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for i, (features, labels) in enumerate(test_loader):
                features = features.unsqueeze(1).float()
                features, labels = features.to(device), labels.to(device).long()
                outputs = model(features)
                predicted = torch.max(outputs.data, 1)[1]
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        accuracy = 100. * correct_test / total_test
        results.append(accuracy)
    
        print(f'Fold {fold}, Test accuracy: {accuracy}%')
    
    # 打印平均準確率
    print(f'Average accuracy: {np.mean(results)}%, std: {np.std(results)}%')
    
    # save model
    data += time.strftime('%d-%H-%M-%S', time.localtime(time.time()))
    torch.save(model.state_dict(), 'D:\Cloud\DLP\lab2\weight\\'+data+'.pth')
    logging.info('Model saved as '+data+'.pth')

if __name__ == '__main__':

    data=['LOSO', 'SD', 'FT']
    train(100,data[0])