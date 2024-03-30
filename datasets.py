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

import os
import urllib.request
import zipfile
import shutil
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class TinyImageNet(Dataset):
    def __init__(self, root="./data", train=True, transform=None, download=True):
        super().__init__()

        self.root = root
        self.train = train
        self.transform = transform

        self.data_dir = os.path.join(self.root, "tiny-imagenet-200")

        if download:
            self.download()

        self.dataset = ImageFolder(root=os.path.join(self.data_dir, "tiny-imagenet-200", "train" if self.train else "val"), transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def download(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        archive_path = os.path.join("/tmp", "tiny-imagenet-200.zip")

        # Download the dataset
        urllib.request.urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip", archive_path)

        # Extract the downloaded archive
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        # Remove the downloaded archive
        os.remove(archive_path)

mean = {
    'MNIST': np.array([0.1307]),
    'FashionMNIST': np.array([0.2859]),
    'cifar10': np.array([0.49139968, 0.48215827, 0.44653124]),
    'cifar10-90': np.array([0.49139968, 0.48215827, 0.44653124]),
    'cifar100': np.array([0.5071, 0.4867, 0.4408]),
    'tinyImagenet': np.array([0.5, 0.5, 0.5])
}
std = {
    'MNIST': [0.3081],
    'FashionMNIST': [0.2859],
    'cifar10': np.array([0.24703233, 0.24348505, 0.26158768]),
    'cifar10-90': np.array([0.24703233, 0.24348505, 0.26158768]),
    'cifar100': np.array([0.2675, 0.2565, 0.2761]),
    'tinyImagenet': np.array([0.5, 0.5, 0.5])
}
dataset_classes = {
    'MNIST': torchvision.datasets.MNIST,
    'FashionMNIST': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar10-90': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
    'tinyImagenet': TinyImageNet,
}


def get_dataloaders(dataset_name, batch_size, seed, image_size, val_frac=0.2, num_workers=1):
    train_transforms = {
        'MNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
        'FashionMNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
        'cifar10': [transforms.Resize((image_size, image_size)),  # resize the image
                    transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                    transforms.RandomRotation(10),  # Rotates the image
                    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    # Perform actions like zooms, change shear angles.
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                    ],
        'cifar10-90': [transforms.RandomCrop(image_size, padding=4),
                       transforms.RandomHorizontalFlip(),
                       ],
        'cifar100': [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                     ],
        'tinyImagenet':[
            # transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                     ],


    }
    input_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'cifar10': [3, image_size, image_size],
                   'cifar10-90': [3, image_size, image_size], 'cifar100': [3, 32, 32], 'tinyImagenet': [3, 32, 32]}



    dataset_class = dataset_classes[dataset_name]
    default_transform = [transforms.Resize((image_size, image_size)),
                         transforms.ToTensor(),
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
                                 batch_size=batch_size, num_workers=num_workers, sampler=None, shuffle=True, pin_memory=True)
    testDataloader = DataLoader(testDataset,
                                batch_size=batch_size, num_workers=num_workers, sampler=None, shuffle=True, pin_memory=True)
    valDataloader = DataLoader(valDataset,
                               batch_size=batch_size, num_workers=num_workers, sampler=None, shuffle=True, pin_memory=True)

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
