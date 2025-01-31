import os
import pickle
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer


class CIVILCOMMENTS(Dataset):
    def __init__(self, x, y, p, isTrain, target_resolution=None):
        self.x = x
        self.y_array = y
        self.p_array = p
        self.isTrain = isTrain
        self.target_resolution = target_resolution
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        x = self.x[idx]
        x = self.tokenizer(x, padding="max_length", truncation=True, max_length=220, return_tensors="pt")
        x = torch.squeeze(torch.stack((x["input_ids"], x["attention_mask"], x["token_type_ids"]), dim=2), dim=0)

        y = self.y_array[idx]
        p = self.p_array[idx]
        return x, y, p, idx



def get_civilcomments(data_dir):
    save_dir = os.path.join(data_dir, f"multinli.pkl")
    if os.path.exists(save_dir):
        print("Loading Dataset")
        with open(save_dir, 'rb') as f:
            X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test = pickle.load(f)
        return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test

    # Meta Data
    df = pd.read_csv(
        os.path.join(data_dir, "all_data_with_identities.csv"),
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
    df = df.rename(columns={"no active attributes": "a"})
    cols_to_keep = ["comment_text", "split", "y", "a"]
    df = df[cols_to_keep]

    df_subset = df[df["split"] == "train"]
    X_train = df_subset["comment_text"].tolist()
    Y_train = np.array(df_subset["y"].tolist())
    P_train = np.array(df_subset["a"].tolist())

    df_subset = df[df["split"] == "val"]
    X_val = df_subset["comment_text"].tolist()
    Y_val = np.array(df_subset["y"].tolist())
    P_val = np.array(df_subset["a"].tolist())

    df_subset = df[df["split"] == "test"]
    X_test = df_subset["comment_text"].tolist()
    Y_test = np.array(df_subset["y"].tolist())
    P_test = np.array(df_subset["a"].tolist())

    with open(save_dir, 'wb') as f:
        pickle.dump((X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test), f)
    return X_train, Y_train, P_train, X_val, Y_val, P_val, X_test, Y_test, P_test
