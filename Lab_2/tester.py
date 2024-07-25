# implement your testing script here
import torch
import torch.nn as nn
import Dataloader
import model.SCCNet as SCCNet
import logging
import os
import numpy as np
import argparse
# init logger
logging.basicConfig(level=logging.DEBUG)

def test(data,model_dic,numClasses, timeSample, Nu, C, Nc, Nt, dropoutRate,lr,weight_decay,scheduler,batch_size,padding1,padding2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)

    # load dataset
    test_dataset = Dataloader.MIBCI2aDataset(mode='test',data=data)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    # load model
    model = SCCNet.SCCNet(numClasses, timeSample, Nu, C, Nc, Nt, dropoutRate)
    model.load_state_dict(torch.load(model_dic))
    model.to(device)
    model.eval()
    loss_temp=[]
    pred_temp=[]
    labels_temp=[]
    # start testing
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.unsqueeze(1).float()
            features, labels = features.to(device), labels.to(device).to(device).long()
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            pred_temp.append(predicted.cpu().numpy())
            labels_temp.append(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss_temp.append(loss.item())
    logging.info('Test Accuracy: %.3f', 100 * correct / total)
    logging.info('Test Loss: %.3f', np.mean(loss_temp))
    return 100 * correct / total, np.mean(loss_temp),pred_temp,labels_temp


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--data", type=str, default='LOSO')
    argparse.add_argument("--model", type=str, default="D:\Cloud\DLP\lab2\weight\\best\LOSO07-19-22-10-31.pth")
    argparse.add_argument("--numClasses", type=int, default=4)
    argparse.add_argument("--timeSample", type=int, default=438)
    argparse.add_argument("--Nu", type=int, default=22)
    argparse.add_argument("--C", type=int, default=22)
    argparse.add_argument("--Nc", type=int, default=20)
    argparse.add_argument("--Nt", type=int, default=1)
    argparse.add_argument("--dropoutRate", type=float, default=0.5)
    argparse.add_argument("--lr", type=float, default=0.001)
    argparse.add_argument("--weight_decay", type=float, default=0.001)
    argparse.add_argument("--scheduler", type=str, default=None)
    argparse.add_argument("--batch_size", type=int, default=512)
    argparse.add_argument("--padding1", default=(0,0))
    argparse.add_argument("--padding2", default=(0,5))
    args = argparse.parse_args()
    test(args.data,args.model ,args.numClasses, args.timeSample, args.Nu, args.C, args.Nc, args.Nt, args.dropoutRate,args.lr,args.weight_decay,args.scheduler,args.batch_size,args.padding1,args.padding2)