# Code is taken from https://github.com/harshays/simplicitybiaspitfalls

import os
import random as rnd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import pickle
from PIL import Image

class MCDOMINOES(Dataset):
    def __init__(self, x, y, p, isTrain, target_resolution=(512, 256)):
        self.x = x
        self.y_array = y
        self.p_array = p
        self.isTrain = isTrain
        self.target_resolution = target_resolution
        self.resize_transform = transforms.Resize(target_resolution)
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(target_resolution, padding=32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.test_transform = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        p = self.p_array[idx]

        img = self.x[idx]
        img = self.resize_transform(img)
        if self.isTrain:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        return img, y, p, idx

    def __getimage__(self, idx):
        img = self.x[idx]
        img = self.resize_transform(img)
        img = np.transpose(np.float32(img), (1, 2, 0))
        img = Image.fromarray(np.uint8(img * 255)).convert('RGB') # h, w, c, 0-255
        return img


def format_mnist(imgs):
    imgs = np.stack([np.pad(imgs[i, 0], 2, constant_values=0) for i in range(len(imgs))])
    imgs = np.expand_dims(imgs, axis=1)
    imgs = np.repeat(imgs, 3, axis=1)
    return torch.tensor(imgs)


def get_mcdominoes(data_dir, spurious_strength, seed):
    VAL_SIZE = 1000
    TEST_SIZE = 2000
    save_dir = os.path.join(data_dir, f"mcdominoes-{spurious_strength}-{seed}.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test = pickle.load(f)
        return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test

    print("Generating Dataset")
    # Load mnist train
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist_train_set = torchvision.datasets.MNIST(os.path.join(data_dir, "mnist"), train=True, download=True)
    mnist_train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()/255.0
    mnist_train_target = mnist_train_set.targets
    rand_perm = torch.randperm(len(mnist_train_input))
    mnist_train_input = mnist_train_input[rand_perm]
    mnist_train_target = mnist_train_target[rand_perm]

    # Load mnist test
    mnist_test_set = torchvision.datasets.MNIST(os.path.join(data_dir, "mnist"), train=False, download=True)
    mnist_test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()/255.0
    mnist_test_target = mnist_test_set.targets

    # Load cifar train
    cifar_train_set = torchvision.datasets.CIFAR10(os.path.join(data_dir, "cifar10"), train=True, download=True, transform=transform)
    cifar_train_input = []
    cifar_train_target = []
    for x, y in cifar_train_set:
        cifar_train_input.append(x)
        cifar_train_target.append(y)
    cifar_train_input = torch.stack(cifar_train_input)
    cifar_train_target = torch.Tensor(cifar_train_target)
    rand_perm = torch.randperm(len(cifar_train_input))
    cifar_train_input = cifar_train_input[rand_perm]
    cifar_train_target = cifar_train_target[rand_perm]

    # Load cifar test
    cifar_test_set = torchvision.datasets.CIFAR10(os.path.join(data_dir, "cifar10"), train=False, download=True, transform=transform)
    cifar_test_input = []
    cifar_test_target = []
    for x, y in cifar_test_set:
      cifar_test_input.append(x)
      cifar_test_target.append(y)
    cifar_test_input = torch.stack(cifar_test_input)
    cifar_test_target = torch.Tensor(cifar_test_target)

    # The new dataset that concatenates MNIST and CIFAR
    mnist_train_input_new, mnist_train_target_new = [], []
    mnist_val_input_new, mnist_val_target_new = [], []
    mnist_test_input_new, mnist_test_target_new = [], []
    cifar_train_input_new, cifar_train_target_new = [], []
    cifar_val_input_new, cifar_val_target_new = [], []
    cifar_test_input_new, cifar_test_target_new = [], []

    # Sort data based on classes
    for i in range(10):
        # For train and validation
        mnist_class_input = mnist_train_input[torch.where(mnist_train_target==i)]
        mnist_val_input_new.extend(mnist_class_input[:int(VAL_SIZE/10)])
        mnist_val_target_new.extend([i]*int(VAL_SIZE/10))
        assert len(mnist_val_input_new) == len(mnist_val_target_new)

        mnist_train_input_new.extend(mnist_class_input[int(VAL_SIZE/10):4500])
        mnist_train_target_new.extend([i]*(4500-int(VAL_SIZE/10)))
        assert len(mnist_train_input_new) == len(mnist_train_target_new)

        cifar_class_input = cifar_train_input[torch.where(cifar_train_target==i)]
        cifar_val_input_new.extend(cifar_class_input[:int(VAL_SIZE/10)])
        cifar_val_target_new.extend([i]*int(VAL_SIZE/10))
        assert len(cifar_val_input_new) == len(cifar_val_target_new)

        cifar_train_input_new.extend(cifar_class_input[int(VAL_SIZE/10):4500])
        cifar_train_target_new.extend([i]*(4500-int(VAL_SIZE/10)))
        assert len(cifar_train_input_new) == len(cifar_train_target_new)
        assert len(cifar_train_input_new) == len(mnist_train_input_new)
        # For test
        mnist_class_input = mnist_test_input[torch.where(mnist_test_target==i)][:800]
        mnist_test_input_new.extend(mnist_class_input)
        mnist_test_target_new.extend([i]*800)
        assert len(mnist_test_input_new) == len(mnist_test_target_new)

        cifar_class_input = cifar_test_input[torch.where(cifar_test_target==i)][:800]
        cifar_test_input_new.extend(cifar_class_input)
        cifar_test_target_new.extend([i]*800)
        assert len(cifar_test_input_new) == len(cifar_test_target_new)

    # Format MNIST
    mnist_train_input = format_mnist(torch.stack(mnist_train_input_new, dim=0))
    mnist_train_target = torch.tensor(mnist_train_target_new)
    mnist_val_input = format_mnist(torch.stack(mnist_val_input_new, dim=0))
    mnist_val_target = torch.tensor(mnist_val_target_new)
    mnist_test_input = format_mnist(torch.stack(mnist_test_input_new, dim=0))
    mnist_test_target = torch.tensor(mnist_test_target_new)

    # Format CIFAR
    cifar_train_input = torch.stack(cifar_train_input_new, dim=0)
    cifar_train_target = torch.tensor(cifar_train_target_new)
    cifar_val_input = torch.stack(cifar_val_input_new, dim=0)
    cifar_val_target = torch.tensor(cifar_val_target_new)
    cifar_test_input = torch.stack(cifar_test_input_new, dim=0)
    cifar_test_target = torch.tensor(cifar_test_target_new)

    # For train, shuffle fraction of dataset which determines spurious strength
    if spurious_strength != 1:
        fraction_permute = 1 - spurious_strength
        permute_indicies = rnd.sample(list(np.arange(len(cifar_train_input))), int(fraction_permute*len(cifar_train_input)))
        shuffled_indicies = rnd.sample(permute_indicies, len(permute_indicies))
        cifar_train_input[permute_indicies] = torch.clone(cifar_train_input[shuffled_indicies])
        cifar_train_target[permute_indicies] = torch.clone(cifar_train_target[shuffled_indicies])

    X_train = torch.cat((mnist_train_input, cifar_train_input), dim=2)
    P_train = mnist_train_target
    Y_train = cifar_train_target
    # Then shuffle all
    rand_perm = torch.randperm(len(X_train))
    X_train = X_train[rand_perm]
    P_train = np.array(P_train[rand_perm])
    Y_train = np.array(Y_train[rand_perm])

    # For validation and test, shuffle then concatenate to remove spurious correlation
    rand_perm = torch.randperm(len(mnist_val_input))
    mnist_val_input = mnist_val_input[rand_perm]
    mnist_val_target = mnist_val_target[rand_perm]
    rand_perm = torch.randperm(len(cifar_val_input))
    cifar_val_input = cifar_val_input[rand_perm]
    cifar_val_target = cifar_val_target[rand_perm]
    X_val = torch.cat((mnist_val_input, cifar_val_input), dim=2)
    P_val = np.array(mnist_val_target)
    Y_val = np.array(cifar_val_target)

    rand_perm = torch.randperm(len(mnist_test_input))
    mnist_test_input = mnist_test_input[rand_perm]
    mnist_test_target = mnist_test_target[rand_perm]
    rand_perm = torch.randperm(len(cifar_test_input))
    cifar_test_input = cifar_test_input[rand_perm]
    cifar_test_target = cifar_test_target[rand_perm]

    X_test = torch.cat((mnist_test_input, cifar_test_input), dim=2)[:TEST_SIZE]
    P_test = np.array(mnist_test_target)[:TEST_SIZE]
    Y_test = np.array(cifar_test_target)[:TEST_SIZE]

    with open(save_dir, 'wb') as f:
        pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)
    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
