from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
from PIL import Image
import numpy as np
import torch
class iclevr(Dataset):
    def __init__(self, path=None, _transforms=None, mode="train",partial=1.0):
        super().__init__()
        self.path = path
        self.mode = mode
        self.partial = partial
        self.transforms = _transforms
        if mode == 'train':
            with open(path+'/train.json', 'r') as file:
                self.json_data = json.load(file)
            self.img_paths = list(self.json_data.keys())
            self.labels = list(self.json_data.values())
        elif mode == 'test':
            with open(path+'/test.json', 'r') as file:
                self.json_data = json.load(file)
            self.labels = self.json_data
        elif mode == 'new_test':
            with open(path+'/new_test.json', 'r') as file:
                self.json_data = json.load(file)
            self.labels = self.json_data
        self.labels_one_hot = []
        with open(path+'objects.json', 'r') as file:
            self.objects_dict = json.load(file)
        for label in self.labels:
            label_one_hot = [0] * len(self.objects_dict)
            for text in label:
                label_one_hot[self.objects_dict[text]] = 1
            self.labels_one_hot.append(label_one_hot)
        self.labels_one_hot = torch.tensor(np.array(self.labels_one_hot))
        if self.transforms is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                # transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = _transforms
        self.mode = mode
    def __len__(self):
        return int(len(self.labels)*self.partial)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            img = Image.open(self.path+'iclevr/'+self.img_paths[idx]).convert('RGB')
            label = self.labels_one_hot[idx]
            return self.transform(img), label
        elif self.mode == 'test':
            return self.labels_one_hot[idx]
        elif self.mode == 'new_test':
            return self.labels_one_hot[idx]