import pickle
import os
import random
import numpy as np
import torch

from PIL import Image
from torch.utils.data import  Dataset
from torchvision import transforms


class SPAWRIOUS(Dataset):
    def __init__(self, x, y, p, isTrain, target_resolution=(512, 512)):
        self.y_array = y
        self.p_array = p
        self.x = x
        self.target_resolution = target_resolution
        self.isTrain = isTrain
        self.resize_transform = transforms.Resize(target_resolution)
        self.train_transform = transforms.Compose([
                transforms.RandomCrop(target_resolution, padding=32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.test_transform = transforms.Compose([
                transforms.transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        y = self.y_array[index]
        p = self.p_array[index]

        img = self.x[index]
        img = Image.open(img).convert("RGB")
        img = self.resize_transform(img)
        if self.isTrain:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        return img, y, p, index

    def __getimage__(self, idx):
        img = self.x[idx]
        img = self.resize_transform(Image.open(img).convert("RGB"))
        return img

def get_spawrious(data_dir, spurious_strength, seed):
    VAL_SIZE = 1000
    TEST_SIZE = 2000

    save_dir = os.path.join(data_dir, f"spawrious-{spurious_strength}-{seed}.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test = pickle.load(f)
        return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
    print("Generating Dataset")

    class_mapping = {"bulldog": 0, "corgi": 1, "dachshund": 2, "labrador": 3}
    place_mapping = {"jungle": 0, "desert": 1, "mountain": 2, "snow": 3, "beach": 4}

    majority_size = 6330 - VAL_SIZE//8 - TEST_SIZE//8
    minority_size = int(majority_size / spurious_strength * (1-spurious_strength))
    train_combinations = {
        "bulldog-jungle": majority_size,
        "bulldog-mountain": minority_size,
        "corgi-desert": majority_size,
        "corgi-jungle": minority_size,
        "dachshund-mountain": majority_size,
        "dachshund-snow": minority_size,
        "labrador-snow": majority_size,
        "labrador-desert": minority_size}

    X_train, Y_train, P_train = [], [], []
    X_val, Y_val, P_val = [], [], []
    X_test, Y_test, P_test = [], [], []

    for group, count in train_combinations.items():
        cls, place = group.split("-")
        # Dataset is split over 2 directories, which we have to combine
        base_dir_0 = os.path.join(data_dir, f"{0}/{place}/{cls}")
        base_dir_1 = os.path.join(data_dir, f"{1}/{place}/{cls}")
        X = [os.path.join(base_dir_0, x) for x in os.listdir(base_dir_0) if x.endswith((".png", ".jpg", ".jpeg"))] + \
            [os.path.join(base_dir_1, x) for x in os.listdir(base_dir_1) if x.endswith((".png", ".jpg", ".jpeg"))]
        # Check if image can be opened
        X_new = []
        for img in X:
            try:
                Image.open(img).convert("RGB")
            except:
                print(f"{img} is corrupted")
                continue
            X_new.append(img)
        X = X_new
        random.shuffle(X)
        Y = [class_mapping[cls] for _ in X]
        P = [place_mapping[place] for _ in X]

        X_train.extend(X[:count])
        Y_train.extend(Y[:count])
        P_train.extend(P[:count])

        X_val.extend(X[count: count+VAL_SIZE//8])
        Y_val.extend(Y[count: count+VAL_SIZE//8])
        P_val.extend(P[count: count+VAL_SIZE//8])

        X_test.extend(X[count+VAL_SIZE//8:count+VAL_SIZE//8+TEST_SIZE//8])
        Y_test.extend(Y[count+VAL_SIZE//8:count+VAL_SIZE//8+TEST_SIZE//8])
        P_test.extend(P[count+VAL_SIZE//8:count+VAL_SIZE//8+TEST_SIZE//8])

    Y_train, P_train, Y_val, P_val, Y_test, P_test = np.array(Y_train), np.array(P_train), \
                                                                   np.array(Y_val), np.array(P_val), \
                                                                   np.array(Y_test), np.array(P_test)
    rand_perm = torch.randperm(len(X_train))
    X_train = [X_train[i] for i in rand_perm]
    P_train = np.array(P_train[rand_perm])
    Y_train = np.array(Y_train[rand_perm])

    with open(save_dir, 'wb') as f:
        pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)

    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test