'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision
from PIL import Image


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    dataset_name = None  # additional parameter: specify the dataset name
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dataset_name = dName  # name of the dataset
        self.train_loader = None
        self.test_loader = None
    
    def load(self):
        print(f'loading {self.dataset_name} dataset...')

        # load dataset
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            loaded_dataset = pickle.load(f)

        # separate train and test dataset
        train_images = np.array([instance['image'] for instance in loaded_dataset['train']])
        train_labels = np.array([instance['label'] for instance in loaded_dataset['train']])
        test_images = np.array([instance['image'] for instance in loaded_dataset['test']])
        test_labels = np.array([instance['label'] for instance in loaded_dataset['test']])

        # ensure ORL dataset labels start from 0
        if self.dataset_name == 'ORL':
            train_labels = train_labels - 1  # convert labels from 1-40 to 0-39
            test_labels = test_labels - 1
            print(f"ORL labels range: {train_labels.min()} to {train_labels.max()}")

        if self.dataset_name == 'CIFAR':
            # train dataset use data augmentation
            train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),  # random crop
                transforms.RandomHorizontalFlip(),     # random horizontal flip
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # test dataset use basic transformation
            test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            train_dataset = CustomCIFARDataset(train_images, train_labels, transform=train_transform)
            test_dataset = CustomCIFARDataset(test_images, test_labels, transform=test_transform)
            
            # create data loader
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=128, 
                shuffle=True,
                num_workers=2
            )
            
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=128, 
                shuffle=False,  # test dataset not shuffle
                num_workers=2
            )

        elif self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = CustomMNISTDataset(train_images, train_labels, transform=transform)
            test_dataset = CustomMNISTDataset(test_images, test_labels, transform=transform)
            
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=128, 
                shuffle=True,
                num_workers=2
            )
            
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=128, 
                shuffle=False,
                num_workers=2
            )

        elif self.dataset_name == 'ORL':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = CustomORLDataset(train_images, train_labels, transform=transform)
            test_dataset = CustomORLDataset(test_images, test_labels, transform=transform)
            
            self.train_loader = DataLoader(
                train_dataset, 
                batch_size=128, 
                shuffle=True,
                num_workers=2
            )
            
            self.test_loader = DataLoader(
                test_dataset, 
                batch_size=128, 
                shuffle=False,
                num_workers=2
            )

        print(f"Training set size: {len(train_images)}")
        print(f"Testing set size: {len(test_images)}")
        
        return {'train_loader': self.train_loader, 'test_loader': self.test_loader}

    
class CustomCIFARDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CustomORLDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # if 3-channel image, only take the first channel
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, 0]
        # ensure it is 2D image
        image = np.expand_dims(image, axis=2)
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class CustomMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label