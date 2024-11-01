import os
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd


class CELEBA(Dataset):
    def __init__(self, x, y, p, isTrain, target_resolution=(224, 224)):
        self.x = x
        self.y_array = y
        self.p_array = p
        self.isTrain = isTrain
        self.target_resolution = target_resolution
        self.resize_transform = transforms.Resize(target_resolution)

        self.train_transform = transforms.Compose([
                transforms.RandomCrop(target_resolution, padding=32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        p = self.p_array[idx]
        img = self.x[idx]
        img = self.resize_transform(Image.open(img).convert("RGB"))
        if self.isTrain:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        return img, y, p, idx

    def __getimage__(self, idx):
        img = self.x[idx]
        img = self.resize_transform(Image.open(img).convert("RGB"))
        return img


def get_celeba(data_dir, seed):
    # Random seed does not affect data generation
    save_dir = os.path.join(data_dir, f"celeba.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test = pickle.load(f)
        return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
    print("Generating Dataset")

    with open(os.path.join(data_dir, "celeba/list_eval_partition.txt"), "r") as f:
        splits = f.readlines()
    with open(os.path.join(data_dir, "celeba/list_attr_celeba.txt"), "r") as f:
        attrs = f.readlines()[2:]
    f = open(os.path.join(data_dir, "metadata_celeba.csv"), "w")
    f.write("id,filename,split,y,a\n")
    for i, (split, attr) in enumerate(zip(splits, attrs)):
        fi, si = split.strip().split()
        ai = attr.strip().split()[1:]
        yi = 1 if ai[9] == "1" else 0
        gi = 1 if ai[20] == "1" else 0
        f.write("{},{},{},{},{}\n".format(i + 1, fi, si, yi, gi))
    f.close()
    root = os.path.join(data_dir, "celeba/img_align_celeba/")
    metadata = os.path.join(data_dir, "metadata_celeba.csv")
    df = pd.read_csv(metadata)
    # Train dataset
    df_subset = df[df["split"] == 0]
    X_train = df_subset["filename"].astype(str).map(lambda x: os.path.join(data_dir, root, x)).tolist()
    Y_train = np.array(df_subset["y"].tolist())
    P_train = np.array(df_subset["a"].tolist())
    # Validation dataset
    df_subset = df[df["split"] == 1]
    X_val = df_subset["filename"].astype(str).map(lambda x: os.path.join(data_dir, root, x)).tolist()
    Y_val = np.array(df_subset["y"].tolist())
    P_val = np.array(df_subset["a"].tolist())
    # Test dataset
    df_subset = df[df["split"] == 2]
    X_test = df_subset["filename"].astype(str).map(lambda x: os.path.join(data_dir, root, x)).tolist()
    Y_test = np.array(df_subset["y"].tolist())
    P_test = np.array(df_subset["a"].tolist())

    with open(save_dir, 'wb') as f:
        pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)

    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
