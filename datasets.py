import torchvision
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    Subset,
    DataLoader,
    RandomSampler,
)
import numpy as np
from torchvision import transforms


def get_dataloaders(dataset_name, batch_size):
    mean = {
        'MNIST': np.array([0.1307]),
        'FashionMNIST': np.array([0.2859])
    }
    std = {
        'MNIST': 0.3081,
        'FashionMNIST': 0.2859
    }
    train_transforms = {
        'MNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
        'FashionMNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')]
    }

    dataset_class = torchvision.datasets.FashionMNIST if dataset_name == "FashionMNIST" else torchvision.datasets.MNIST
    default_transform = [transforms.ToTensor(),
                         transforms.Normalize(mean[dataset_name], [std[dataset_name]] * len(mean[dataset_name]))]

    train_transform = transforms.Compose(train_transforms[dataset_name] + default_transform)
    test_transform = transforms.Compose(default_transform)
    # build data sets
    trainDataset = dataset_class(root="./data", train=True, transform=train_transform, download=True)
    testDataset = dataset_class(root="./data", train=False, transform=test_transform, download=True)

    # build data loaders
    trainDataloader = DataLoader(trainDataset,
                                 batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True)

    testDataloader = DataLoader(testDataset,
                                batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True)
    return trainDataloader, testDataloader