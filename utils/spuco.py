import os
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch


class SPUCO(Dataset):
    def __init__(self, x, y, p, isTrain, target_resolution=(512, 512)):
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


def get_waterbirds(data_dir, spurious_strength, split):
    data_dir = os.path.join(data_dir, "spuco_birds")
    LANDBIRDS = "landbirds"
    WATERBIRDS = "waterbirds"
    LAND = "land"
    WATER = "water"
    MAJORITY_SIZE = {
        "train": 4500, # 10,000
        "test": 500, # 500
    }
    MINORITY_SIZE = {
        "train": int(4500 / spurious_strength * (1-spurious_strength)), # 500
        "test": 500, # 500
    }
    X_train, Y_train, P_train = [], [], []
    landbirds_land = os.listdir(os.path.join(data_dir, split, f"{LANDBIRDS}/{LAND}"))[:MAJORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{LANDBIRDS}/{LAND}", x)) for x in landbirds_land])
    Y_train.extend([0] * len(landbirds_land))
    P_train.extend([0] * len(landbirds_land))

    # Landbirds Water
    landbirds_water = os.listdir(os.path.join(data_dir, split, f"{LANDBIRDS}/{WATER}"))[:MINORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{LANDBIRDS}/{WATER}", x)) for x in landbirds_water])
    Y_train.extend([0] * len(landbirds_water))
    P_train.extend([1] * len(landbirds_water))

    # Waterbirds Land
    waterbirds_land = os.listdir(os.path.join(data_dir, split, f"{WATERBIRDS}/{LAND}"))[:MINORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{WATERBIRDS}/{LAND}", x)) for x in waterbirds_land])
    Y_train.extend([1] * len(waterbirds_land))
    P_train.extend([0] * len(waterbirds_land))

    # Waterbirds Water
    waterbirds_water = os.listdir(os.path.join(data_dir, split, f"{WATERBIRDS}/{WATER}"))[:MAJORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{WATERBIRDS}/{WATER}", x)) for x in waterbirds_water])
    Y_train.extend([1] * len(waterbirds_water))
    P_train.extend([1] * len(waterbirds_water))
    return X_train, Y_train, P_train


def get_dogs(data_dir, spurious_strength, split):
    data_dir = os.path.join(data_dir, "spuco_dogs")
    SMALL_DOGS = "small_dogs"
    BIG_DOGS = "big_dogs"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    MAJORITY_SIZE = {
        "train": 4500, # 10,000
        "test": 500, # 500
    }
    MINORITY_SIZE = {
        "train": int(4500 / spurious_strength * (1-spurious_strength)), # 500
        "test": 500, # 500
    }

    X_train, Y_train, P_train = [], [], []
    small_dogs_indoor = os.listdir(os.path.join(data_dir, split, f"{SMALL_DOGS}/{INDOOR}"))[:MAJORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{SMALL_DOGS}/{INDOOR}", x)) for x in small_dogs_indoor])
    Y_train.extend([0] * len(small_dogs_indoor))
    P_train.extend([0] * len(small_dogs_indoor))

    # Small Dogs - Outdoor
    small_dogs_outdoor = os.listdir(os.path.join(data_dir, split, f"{SMALL_DOGS}/{OUTDOOR}"))[:MINORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{SMALL_DOGS}/{OUTDOOR}", x)) for x in small_dogs_outdoor])
    Y_train.extend([0] * len(small_dogs_outdoor))
    P_train.extend([1] * len(small_dogs_outdoor))

    # Big Dogs - Indoor
    big_dogs_indoor = os.listdir(os.path.join(data_dir, split, f"{BIG_DOGS}/{INDOOR}"))[:MINORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{BIG_DOGS}/{INDOOR}", x)) for x in big_dogs_indoor])
    Y_train.extend([1] * len(big_dogs_indoor))
    P_train.extend([0] * len(big_dogs_indoor))

    # Big Dogs - Outdoor
    big_dogs_outdoor = os.listdir(os.path.join(data_dir, split, f"{BIG_DOGS}/{OUTDOOR}"))[:MAJORITY_SIZE[split]]
    X_train.extend([str(os.path.join(data_dir, split, f"{BIG_DOGS}/{OUTDOOR}", x)) for x in big_dogs_outdoor])
    Y_train.extend([1] * len(big_dogs_outdoor))
    P_train.extend([1] * len(big_dogs_outdoor))
    return X_train, Y_train, P_train


def get_spuco(data_dir, spurious_strength, seed):
    VAL_SIZE = 1000
    TEST_SIZE = 2000

    save_dir = os.path.join(data_dir, f"spuco-{spurious_strength}-{seed}.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test = pickle.load(f)
        return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test

    print("Generating Dataset")
    # For waterbirds dataset
    X_train, Y_train, P_train = get_waterbirds(data_dir, spurious_strength, "train")
    X_test, Y_test, P_test = get_waterbirds(data_dir, spurious_strength, "test")

    X_train_dog, Y_train_dog, P_train_dog = get_dogs(data_dir, spurious_strength, "train")
    X_test_dog, Y_test_dog, P_test_dog = get_dogs(data_dir, spurious_strength, "test")

    X_train.extend(X_train_dog)
    Y_train.extend([i + 2 for i in Y_train_dog])
    P_train.extend([i + 2 for i in P_train_dog])

    X_test.extend(X_test_dog)
    Y_test.extend([i + 2 for i in Y_test_dog])
    P_test.extend([i + 2 for i in P_test_dog])

    Y_train, P_train, Y_test, P_test = np.array(Y_train), np.array(P_train), np.array(Y_test), np.array(P_test)
    rand_perm = torch.randperm(len(X_train))
    X_train = [X_train[i] for i in rand_perm]
    P_train = np.array(P_train[rand_perm])
    Y_train = np.array(Y_train[rand_perm])

    # We do not load the validation dataset directly since it is not balanced
    random_idxs = np.arange(len(X_test), dtype=int)
    np.random.shuffle(random_idxs)
    X_val, Y_val, P_val = [X_test[i] for i in random_idxs[:VAL_SIZE]], Y_test[random_idxs[:VAL_SIZE]], P_test[random_idxs[:VAL_SIZE]]
    X_test, Y_test, P_test = [X_test[i] for i in random_idxs[VAL_SIZE:VAL_SIZE+TEST_SIZE]], Y_test[random_idxs[VAL_SIZE:VAL_SIZE+TEST_SIZE]], P_test[random_idxs[VAL_SIZE:VAL_SIZE+TEST_SIZE]]


    with open(save_dir, 'wb') as f:
        pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)
    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test