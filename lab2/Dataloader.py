import torch
import numpy as np
import os
import logging

#init logger
# logger = logging.getLogger(__name__)
class MIBCI2aDataset(torch.utils.data.Dataset):
    def _getFeatures(self, filePath):
        # implement the getFeatures method
        """
        read all the preprocessed data from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        os.chdir(filePath)
        files = os.listdir()
        files.sort()
        features = []
        # logger.info('loading file: %s', filePath)
        for file in files:
            data = np.load(file)
            features.append(data)
        features = np.concatenate(features, axis=0)
        # logger.info('features shape: %s', features.shape)
        return torch.tensor(features)


    def _getLabels(self, filePath):
        # implement the getLabels method
        """
        read all the preprocessed labels from the file path, read it using np.load,
        and concatenate them into a single numpy array
        """
        os.chdir(filePath)
        files = os.listdir()
        files.sort()
        labels = []
        # logger.info('loading file: %s', filePath)
        for file in files:
            data = np.load(file)
            labels.append(data)
        labels = np.concatenate(labels, axis=0)
        # logger.info('labels shape: %s', labels.shape)
        return torch.tensor(labels)

    def __init__(self, mode,data):
        # remember to change the file path according to different experiments
        assert mode in ['train', 'test', 'finetune']
        if data == 'LOSO':
            train_data_name = 'LOSO_train'
            test_data_name = 'LOSO_test'
        elif data == 'SD':
            train_data_name = 'SD_train'
            test_data_name = 'SD_test'
        elif data == 'FT':
            test_data_name = 'LOSO_test'

        if mode == 'train' and data != 'FT':
            # subject dependent: ./dataset/SD_train/features/ and ./dataset/SD_train/labels/
            # leave-one-subject-out: ./dataset/LOSO_train/features/ and ./dataset/LOSO_train/labels/
            self.features = self._getFeatures(filePath='D:/Cloud/DLP/lab2/dataset/'+train_data_name+'/features/')
            self.labels = self._getLabels(filePath='D:/Cloud/DLP/lab2/dataset/'+train_data_name+'/labels/')
        if mode == 'finetune' and data == 'FT':
            # finetune: ./dataset/FT/features/ and ./dataset/FT/labels/
            self.features = self._getFeatures(filePath='D:/Cloud/DLP/lab2/dataset/FT/features/')
            self.labels = self._getLabels(filePath='D:/Cloud/DLP/lab2/dataset/FT/labels/')
        if mode == 'test' :
            # subject dependent: ./dataset/SD_test/features/ and ./dataset/SD_test/labels/
            # leave-one-subject-out and finetune: ./dataset/LOSO_test/features/ and ./dataset/LOSO_test/labels/
            self.features = self._getFeatures(filePath='D:/Cloud/DLP/lab2/dataset/'+test_data_name+'/features/')
            self.labels = self._getLabels(filePath='D:/Cloud/DLP/lab2/dataset/'+test_data_name+'/labels/')

    def __len__(self):
        # implement the len method
        length = len(self.features)
        return length

    def __getitem__(self, idx):
        # implement the getitem method
        return self.features[idx], self.labels[idx]