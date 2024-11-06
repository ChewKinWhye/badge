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
    df = pd.read_csv(
        os.path.join(data_dir, "civilcomments", "all_data_with_identities.csv"),
        index_col=0,
    )
    group_attrs = [
        "male",
        "female",
        "LGBTQ",
        "christian",
        "muslim",
        "other_religions",
        "black",
        "white",
    ]
    cols_to_keep = ["comment_text", "split", "toxicity"]
    df = df[cols_to_keep + group_attrs]
    df = df.rename(columns={"toxicity": "y"})
    df["y"] = (df["y"] >= 0.5).astype(int)
    df[group_attrs] = (df[group_attrs] >= 0.5).astype(int)
    df["no active attributes"] = 0
    df.loc[(df[group_attrs].sum(axis=1)) == 0, "no active attributes"] = 1
    few_groups, all_groups = [], []
    train_df = df.groupby("split").get_group("train")
    split_df = train_df.rename(columns={"no active attributes": "a"})
    few_groups.append(split_df[["y", "split", "comment_text", "a"]])
    for split, split_df in df.groupby("split"):
        for i, attr in enumerate(group_attrs):
            test_df = split_df.loc[
                split_df[attr] == 1, ["y", "split", "comment_text"]
            ].copy()
            test_df["a"] = i
            all_groups.append(test_df)
            if split != "train":
                few_groups.append(test_df)
    few_groups = pd.concat(few_groups).reset_index(drop=True)
    all_groups = pd.concat(all_groups).reset_index(drop=True)
    for name, df in {"coarse": few_groups, "fine": all_groups}.items():
        df.index.name = "filename"
        df = df.reset_index()
        df["id"] = df["filename"]
        df["split"] = df["split"].replace({"train": 0, "val": 1, "test": 2})
        text = df.pop("comment_text")

        df[["id", "filename", "split", "y", "a"]].to_csv(
            os.path.join(data_dir, f"metadata_civilcomments_{name}.csv"), index=False
        )
        text.to_csv(
            os.path.join(data_dir, "civilcomments", f"civilcomments_{name}.csv"),
            index=False,
        )

    # with open(save_dir, 'wb') as f:
    #     pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)
    # return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
