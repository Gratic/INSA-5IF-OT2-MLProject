import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from sklearn.model_selection import train_test_split
from .torchsampler import ImbalancedDatasetSampler
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision import get_image_backend

def get_transform():
    return transforms.Compose(
    [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
     transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
     transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

def get_augmented_transform():
    return transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(degrees=(0, 359)),
     ])

def get_both_transform():
    return transforms.Compose(
    [
        transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
        transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
        transforms.Normalize(mean=(0.5,),std=(0.5,)), # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 359)),
     ]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

current_absolute_path = os.path.dirname(__file__)

train_dir = current_absolute_path + '/train_images'
test_dir = current_absolute_path + '/test_images'

transform = get_transform()
transform_augment = get_augmented_transform()
both_transform = get_both_transform()

def basic_load(valid_size = 0.2, batch_size = 32, device = 'cpu'): # proportion of validation set (80% train, 20% validation)
    # Define two pytorch datasets (train/test) 
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    # Define randomly the indices of examples to use for training and for validation
    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

    # Define two "samplers" that will randomly pick examples from the training and validation set
    train_sampler = SubsetRandomSampler(train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Dataloaders (take care of loading the data from disk, batch by batch, during training)
    kwargs = {'num_workers': 4, 'pin_memory': True} if 'cuda' in device else {}
    print(kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

    classes = ('noface','face')  # indicates that "1" means "face" and "0" non-face (only used for display)

    # Training 

    # loop over epochs: one epoch = one pass through the whole training dataset
    # for epoch in range(1, n_epochs+1):  
    #   loop over iterations: one iteration = 1 batch of examples
    #   for data, target in train_loader: 

    return (train_loader, valid_loader, test_loader, classes)

def imbalanced_load(valid_size = 0.2, batch_size = 32, device = 'cpu'): # proportion of validation set (80% train, 20% validation)
    # Define two pytorch datasets (train/test) 
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=both_transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    # Define randomly the indices of examples to use for training and for validation
    num_train = len(train_data)
    indices_train = list(range(num_train))
    np.random.shuffle(indices_train)
    split_tv = int(np.floor(valid_size * num_train))
    train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

    # Define two "samplers" that will randomly pick examples from the training and validation set
    train_sampler = ImbalancedDatasetSampler(train_data, train_new_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Dataloaders (take care of loading the data from disk, batch by batch, during training)
    kwargs = {'num_workers': 4, 'pin_memory': True} if 'cuda' in device else {}
    # print(kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

    classes = ('noface','face')  # indicates that "1" means "face" and "0" non-face (only used for display)

    return (train_loader, valid_loader, test_loader, classes, train_data, train_new_idx)

def balanced_load(valid_size = 0.2, batch_size = 32, device = 'cpu'):
    # Define two pytorch datasets (train/test) 
    train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    test_data = torchvision.datasets.ImageFolder(test_dir, transform=transform)

    targets = train_data.targets

    train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=valid_size, shuffle=True, stratify=targets)

    print(len(train_idx))
    print(len(valid_idx))

    # Define two "samplers" that will randomly pick examples from the training and validation set
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Dataloaders (take care of loading the data from disk, batch by batch, during training)
    kwargs = {'num_workers': 4, 'pin_memory': True} if 'cuda' in device else {}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

    classes = ('noface','face')  # indicates that "1" means "face" and "0" non-face (only used for display)

    return (train_loader, valid_loader, test_loader, classes)

class AugmentFacesDataset(torchvision.datasets.ImageFolder):
    def __init__(self,
        root: str,
        transform: Optional[Callable] = None,
        transform_augment: Optional[Callable] = None):
        super().__init__(root, transform)
        self.transform_augment = transform_augment
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.transform_augment is not None and target == 1:
            sample = self.transform_augment(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
