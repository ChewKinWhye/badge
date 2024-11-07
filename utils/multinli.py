import os
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd


class MULTINLI(Dataset):
    def __init__(self, x, y, p, isTrain, target_resolution=None):
        self.x = x
        self.y_array = y
        self.p_array = p
        self.isTrain = isTrain
        self.target_resolution = target_resolution

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y_array[idx]
        p = self.p_array[idx]
        return x, y, p, idx



def get_multinli(data_dir):
    save_dir = os.path.join(data_dir, f"multinli.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test = pickle.load(f)
        return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test

    # Meta Data
    df = pd.read_csv(os.path.join(data_dir, "metadata_random.csv"), index_col=0)
    df = df.rename(columns={"gold_label": "y", "sentence2_has_negation": "a"})
    df = df.reset_index(drop=True)
    df.index.name = "id"
    df = df.reset_index()
    df["filename"] = df["id"]
    df = df.reset_index()[["id", "filename", "split", "y", "a"]]
    df.to_csv(os.path.join(data_dir, "metadata_multinli.csv"), index=False)

    features_array = []
    for feature_file in ["cached_train_bert-base-uncased_128_mnli", "cached_dev_bert-base-uncased_128_mnli", "cached_dev_bert-base-uncased_128_mnli-mm"]:
        features = torch.load(os.path.join(data_dir, feature_file))
        features_array += features
    all_input_ids = torch.tensor([f.input_ids for f in features_array], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_mask for f in features_array], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features_array], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features_array], dtype=torch.long)
    x_array = torch.stack((all_input_ids, all_input_masks, all_segment_ids), dim=2)

    metadata = os.path.join(data_dir, "metadata_multinli.csv")
    df = pd.read_csv(metadata)
    df_subset = df[df["split"] == 0]
    X_train = x_array[df_subset["filename"].tolist()]
    Y_train = np.array(df_subset["y"].tolist())
    P_train = np.array(df_subset["a"].tolist())

    metadata = os.path.join(data_dir, "metadata_multinli.csv")
    df = pd.read_csv(metadata)
    df_subset = df[df["split"] == 1]
    X_val = x_array[df_subset["filename"].tolist()]
    Y_val = np.array(df_subset["y"].tolist())
    P_val = np.array(df_subset["a"].tolist())
    metadata = os.path.join(data_dir, "metadata_multinli.csv")
    df = pd.read_csv(metadata)
    df_subset = df[df["split"] == 2]
    X_test = x_array[df_subset["filename"].tolist()]
    Y_test = np.array(df_subset["y"].tolist())
    P_test = np.array(df_subset["a"].tolist())
    with open(save_dir, 'wb') as f:
        pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)
    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
