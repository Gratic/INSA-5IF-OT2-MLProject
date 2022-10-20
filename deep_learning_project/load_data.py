import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os

current_absolute_path = os.path.dirname(__file__)

train_dir = current_absolute_path + '/train_images'
test_dir = current_absolute_path + '/test_images'

transform = transforms.Compose(
    [transforms.Grayscale(),   # transforms to gray-scale (1 input channel)
     transforms.ToTensor(),    # transforms to Torch tensor (needed for PyTorch)
     transforms.Normalize(mean=(0.5,),std=(0.5,))]) # subtracts mean (0.5) and devides by standard deviation (0.5) -> resulting values in (-1, +1)

def load(valid_size = 0.2, batch_size = 8, device = 'cpu'): # proportion of validation set (80% train, 20% validation)
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
    kwargs = {'num_workers': 4, 'pin_memory': True} if device=='cuda' else {}
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