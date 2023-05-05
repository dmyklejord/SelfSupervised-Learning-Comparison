import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from PIL import Image

import numpy as np


# Set image augmentations. These are the default SimCLR augmentations.
class simclr_transforms():
    size = 224
    s = 1
    color_jitter = torchvision.transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    train = torchvision.transforms.Compose([
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
    ])
    test= torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(size, size)),
                    torchvision.transforms.ToTensor(),
    ])

class ReducedTransforms():
    # Define the transformations to be applied to the input images. resnet-18 needs a 224x224 image.
    train = torchvision.transforms.Compose([
                    torchvision.transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation(90),
                    torchvision.transforms.ToTensor()
    ])

    test = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.ToTensor()
    ])

# # Define the transformations to be applied to the input images. resnet-18 needs a 224x224 image.
# transform = transforms.Compose([
#     transforms.RandomResizedCrop((224,224), scale=(0.4, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(90),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ]) 


class CIFAR10Pair(CIFAR10):
    """
    From: https://github.com/leftthomas/SimCLR/blob/master/utils.py
    A stochastic data augmentation module that transforms
    any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        # Transform for the knn feature bank function:
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def __len__(self):
        return len(self.data)


def get_dataloaders_cifar10(
    train_transforms=simclr_transforms.train,
    test_transforms=simclr_transforms.test,
    batch_size=128,
    num_workers=0):

    train_simclr_dataset = CIFAR10Pair(
        root='Cifar10_data',
        download=True,
        train=True,
        transform=train_transforms)

    test_simclr_dataset = CIFAR10Pair(
        root='Cifar10_data',
        download=True,
        train=False,
        transform=test_transforms)

    train_loader = DataLoader(
        train_simclr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    test_loader = DataLoader(
        test_simclr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    return train_loader, test_loader


class ImagePair(torchvision.datasets.ImageFolder):
    """
    Returns the same image transformed two ways.
    Modified from: https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample1 ,sample2, target) where target is class_index of the target class.
            sample1 and sample2 are augmented/transformed versions of sample (sample==image).
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample1 = self.transform(sample)
            sample2 = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, sample2, target

    def __len__(self):
        return len(self.samples)


def get_dataloaders(
    dir='data',
    train_transforms=simclr_transforms.train,
    test_transforms=simclr_transforms.test,
    batch_size=128,
    test_fraction=0.20,
    num_workers=0):

    # Getting transformed image pairs of the seedling data:
    train_dataset = ImagePair(root=dir, transform=train_transforms)
    test_dataset = ImagePair(root=dir, transform=test_transforms)

    # Splitting into training and testing datasets
    # using the test_fraction split:
    length_data = len(train_dataset)
    dataset_indices = range(length_data)
    length_test = int(test_fraction * length_data)
    train_sampler, test_sampler = random_split(dataset_indices, [length_data-length_test, length_test], generator=torch.Generator().manual_seed(123))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              drop_last=True,
                              sampler=train_sampler)


    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             sampler=test_sampler)

    return train_loader, test_loader

def get_dataloaders_reduced_data(
    dir='data',
    train_transforms=simclr_transforms.train,
    test_transforms=simclr_transforms.test,
    batch_size=128,
    test_fraction=0.20,
    num_datapoints=20,
    num_workers=0):

    # Getting transformed image pairs of the seedling data:
    train_dataset = ImagePair(root=dir, transform=train_transforms)
    test_dataset = ImagePair(root=dir, transform=test_transforms)

    # Splitting into training and testing datasets
    # using the test_fraction split:
    length_data = len(train_dataset)
    dataset_indices = range(length_data)
    length_test = int(test_fraction * length_data)
    
    # The datapoint indices of the original dataset are randomly split into testing and training. 
    train_sampler, test_sampler = random_split(dataset_indices, [length_data-length_test, length_test], generator=torch.Generator().manual_seed(123))

    # Getting num_datapoints from each class:
    train_idxs = []
    for class_idx in range(len(np.unique(train_dataset.targets))):
        
        # Gets the indices of the all the samples of a certain class
        # i.e. all the indicies of the first class
        class_idxs = np.where(np.array(train_dataset.targets) == class_idx)[0]

        # Makes sure that those indices belong to the training set
        random_class_idxs = []
        for idx in class_idxs:
            if idx in train_sampler.indices:
                random_class_idxs.append(idx)

        # Randomly pick out num_datapoints from the class
        train_idxs = np.concatenate((train_idxs, np.random.choice(random_class_idxs, num_datapoints, replace=False)))
    
    reduced_train_dataset = torch.utils.data.Subset(train_dataset, train_idxs.astype(int))
    reduced_test_dataset = torch.utils.data.Subset(test_dataset, test_sampler.indices)

    train_loader = DataLoader(dataset=reduced_train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True)


    test_loader = DataLoader(dataset=reduced_test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=False)

    return train_loader, test_loader
