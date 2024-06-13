from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from PIL import Image

import numpy as np
import torch

def get_dataset(name, path):
    if name == 'MNIST':
        X_tr, Y_tr, X_te, Y_te = get_MNIST(path)
    elif name == 'FashionMNIST':
        X_tr, Y_tr, X_te, Y_te = get_FashionMNIST(path)
    elif name == 'SVHN':
        X_tr, Y_tr, X_te, Y_te = get_SVHN(path)
    elif name == 'CIFAR10':
        X_tr, Y_tr, X_te, Y_te = get_CIFAR10(path)
    else:
        print('Choose a valid dataset function.')
        raise ValueError
    # Split train into train and validation
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_tr, Y_tr, test_size=10000)
    # Convert all datasets to numpy
    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = [np.array(i) for i in [X_tr, Y_tr, X_val, Y_val, X_te, Y_te]]
    return X_tr, Y_tr, X_val, Y_val, X_te, Y_te

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name == 'MNIST':
        return MNIST_Handler
    elif name == 'FashionMNIST':
        return FashionMNIST_Handler
    elif name == 'SVHN':
        return SVHN_Handler
    elif name == 'CIFAR10':
        return CIFAR_Handler

class MNIST_Handler(Dataset):
    def __init__(self, X, Y, is_train):
        self.X = X
        self.Y = Y
        self.is_train = is_train
        if self.is_train:
            self.transform = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class FashionMNIST_Handler(Dataset):
    def __init__(self, X, Y, is_train):
        self.X = X
        self.Y = Y
        self.is_train = is_train
        if self.is_train:
            self.transform = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class SVHN_Handler(Dataset):
    def __init__(self, X, Y, is_train):
        self.X = X
        self.Y = Y
        self.is_train = is_train
        if self.is_train:
            self.transform = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


class CIFAR_Handler(Dataset):
    def __init__(self, X, Y, is_train):
        self.X = X
        self.Y = Y
        self.is_train = is_train

        if self.is_train:
            self.transform = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
