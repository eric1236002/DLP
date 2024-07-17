# implement your testing script here
import torch
import Dataloader
import model.SCCNet as SCCNet
import logging
import os
# init logger
logging.basicConfig(level=logging.DEBUG)

def test(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device: %s', device)

    # load dataset
    test_dataset = Dataloader.MIBCI2aDataset(mode='test',data=data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    # load model
    model = SCCNet.SCCNet(numClasses=4, timeSample=438, Nu=22, C=22, Nc=20, Nt=1, dropoutRate=0.5)
    model.load_state_dict(torch.load('D:\Cloud\DLP\lab2\weight\LOSO18-01-02-49.pth'))
    model.to(device)
    model.eval()

    # start testing
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            features = features.unsqueeze(1).float()
            features, labels = features.to(device), labels.to(device).to(device).long()
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # logging.info('Batch: %d, Accuracy: %.3f', i, 100 * correct / total)

    print('Accuracy: ', 100 * correct / total)

if __name__ == '__main__':
    data=['LOSO', 'SD', 'FT']
    test(data[0])