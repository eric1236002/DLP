# script for drawing figures, and more if needed
import matplotlib.pyplot as plt
import torch

def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total