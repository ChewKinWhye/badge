import numpy as np
import os

from utils.mcdominoes import get_mcdominoes, MCDOMINOES
from utils.spuco import get_spuco, SPUCO
from utils.spawrious import get_spawrious, SPAWRIOUS


def log_data(x, y, p, handler, dataset_name, dataset_split):
    base_dir = os.path.join("data", dataset_name, dataset_split)
    os.makedirs(base_dir, exist_ok=True)
    print(dataset_split)
    unique_y, unique_p = np.unique(y), np.unique(p)
    for y_value in unique_y:
        for p_value in unique_p:
            group_idxs = np.arange(len(y))[np.where((y == y_value) & (p == p_value))]
            print(f"Group Counts for y == {y_value} and p == {p_value}: {len(group_idxs)}")


def get_data(dataset, data_dir, spurious_strength, seed):
    data_dir = os.path.join(data_dir, dataset)
    if dataset == "mcdominoes":
        num_classes = 10
        num_attributes = 10
        handler = MCDOMINOES
        X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te = \
            get_mcdominoes(data_dir, spurious_strength, seed)
        X_tr, X_val, X_te = [i for i in X_tr], [i for i in X_val], [i for i in X_te]
    elif dataset == "spawrious":
        num_classes = 4
        num_attributes = 4
        handler = SPAWRIOUS
        X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te = \
            get_spawrious(data_dir, spurious_strength, seed)
    elif dataset == "spuco":
        num_classes = 4
        num_attributes = 4
        handler = SPUCO
        X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te = \
            get_spuco(data_dir, spurious_strength, seed)
    else:
        print("Data specified not supported")
        exit()

    log_data(X_tr, Y_tr, P_tr, handler, dataset, "Train Dataset")
    log_data(X_val, Y_val, P_val, handler, dataset, "Validation Dataset")
    log_data(X_te, Y_te, P_te, handler, dataset, "Test Dataset")
    return X_tr, Y_tr, P_tr, X_val, Y_val, P_val, X_te, Y_te, P_te, num_classes, num_attributes, handler
