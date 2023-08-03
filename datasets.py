import copy

import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import (
    Dataset,
    ConcatDataset,
    Subset,
    DataLoader,
    RandomSampler,
)
import numpy as np
from torchvision import transforms

mean = {
    'MNIST': np.array([0.1307]),
    'FashionMNIST': np.array([0.2859]),
    'cifar10': np.array([0.49139968, 0.48215827, 0.44653124]),
    'cifar10-90': np.array([0.49139968, 0.48215827, 0.44653124]),
    'cifar100': np.array([0.5071, 0.4867, 0.4408])
}
std = {
    'MNIST': [0.3081],
    'FashionMNIST': [0.2859],
    'cifar10': np.array([0.24703233, 0.24348505, 0.26158768]),
    'cifar10-90': np.array([0.24703233, 0.24348505, 0.26158768]),
    'cifar100': np.array([0.2675, 0.2565, 0.2761])
}
dataset_classes = {
    'MNIST': torchvision.datasets.MNIST,
    'FashionMNIST': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar10-90': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}
train_transforms = {
    'MNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
    'FashionMNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
    'cifar10': [transforms.Resize((32, 32)),  # resize the image
                transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                transforms.RandomRotation(10),  # Rotates the image
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                # Perform actions like zooms, change shear angles.
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                ],
    'cifar10-90': [transforms.RandomCrop(32, padding=4),
                   transforms.RandomHorizontalFlip(),
                   ],
    'cifar100': [transforms.Resize((32, 32)),  # resize the image
                 transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                 transforms.RandomRotation(10),  # Rotates the image
                 transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                 # Perform actions like zooms, change shear angles.
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                 ],

}
input_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'cifar10': [3, 32, 32], 'cifar10-90': [3, 32, 32],
               'cifar100': [3, 32, 32]}


def get_dataloaders(dataset_name, batch_size, seed, image_size, val_frac=0.2):
    dataset_class = dataset_classes[dataset_name]
    default_transform = [transforms.ToTensor(),
                         transforms.Normalize(mean[dataset_name], std[dataset_name])]

    train_transform = transforms.Compose(train_transforms[dataset_name] + default_transform)
    test_transform = transforms.Compose(default_transform)
    # build data sets
    trainDataset = dataset_class(root="./data", train=True, transform=train_transform, download=True)
    valDataset = dataset_class(root="./data", train=True, transform=test_transform, download=True)
    testDataset = dataset_class(root="./data", train=False, transform=test_transform, download=True)

    # split data between train and val
    num_train = len(trainDataset)
    indices = list(range(num_train))
    split = int(np.floor(val_frac * num_train))
    np.random.seed(seed)
    np.random.shuffle(indices)

    if val_frac != 0:
        train_idx, valid_idx = indices[split:], indices[:split]
        trainDataset = torch.utils.data.Subset(trainDataset, train_idx)
        valDataset = torch.utils.data.Subset(valDataset, valid_idx)
    else:
        valDataset = dataset_class(root="./data", train=False, transform=test_transform, download=True)

    # build data loaders
    trainDataloader = DataLoader(trainDataset,
                                 batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True)
    testDataloader = DataLoader(testDataset,
                                batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True)
    valDataloader = DataLoader(valDataset,
                               batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True)

    return trainDataloader, valDataloader, testDataloader, input_shape[dataset_name]


def get_subset(dataset, labels):
    idx = np.in1d(dataset.targets, labels)
    splited_dataset = copy.deepcopy(dataset)
    splited_dataset.targets = splited_dataset.targets[idx]
    splited_dataset.targets = (splited_dataset.targets != labels[0])
    splited_dataset.data = splited_dataset.data[idx]
    for i, (x, y) in enumerate(splited_dataset):
        plt.imshow(x[0])
        plt.title(f"class={y}, task={labels}")
        plt.show()
        if i > 7:
            break
    return splited_dataset


def get_dataloaders_forgetting(dataset_name, batch_size, seed):
    dataset_class = dataset_classes[dataset_name]
    default_transform = [transforms.ToTensor(),
                         transforms.Normalize(mean[dataset_name], std[dataset_name])]

    train_transform = transforms.Compose(train_transforms[dataset_name] + default_transform)
    test_transform = transforms.Compose(default_transform)
    # build data sets
    trainDataset = dataset_class(root="./data", train=True, transform=train_transform, download=True)
    valDataset = dataset_class(root="./data", train=True, transform=test_transform, download=True)
    testDataset = dataset_class(root="./data", train=False, transform=test_transform, download=True)

    # split data between train and val
    num_train = len(trainDataset)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    np.random.seed(seed)
    np.random.shuffle(indices)
    task_permutation = [i for i in range(10)]
    np.random.shuffle(task_permutation)

    train_idx, valid_idx = indices[split:], indices[:split]
    trainDataset.targets = trainDataset.targets[train_idx]
    trainDataset.data = trainDataset.data[train_idx]
    valDataset.targets = valDataset.targets[valid_idx]
    valDataset.data = valDataset.data[valid_idx]

    testDatasetTasks = [get_subset(testDataset, task_permutation[2 * i:2 * i + 2]) for i in range(5)]
    trainDatasetTasks = [get_subset(trainDataset, task_permutation[2 * i:2 * i + 2]) for i in range(5)]
    valDatasetTasks = [get_subset(valDataset, task_permutation[2 * i:2 * i + 2]) for i in range(5)]

    # build data loaders
    trainDataloader = [DataLoader(trainDatasetTasks[i],
                                  batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True) for
                       i in range(5)]

    testDataloader = [DataLoader(testDatasetTasks[i],
                                 batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True) for
                      i in range(5)]
    valDataloader = [DataLoader(valDatasetTasks[i],
                                batch_size=batch_size, num_workers=1, sampler=None, shuffle=True, pin_memory=True) for
                     i in range(5)]
    return trainDataloader, valDataloader, testDataloader, input_shape[dataset_name]
