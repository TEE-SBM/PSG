import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("..")
from src.utils.io import var_to_str, get_path





def get_iterates(objective, optimizer, dataset="amazon"):
    model_cfg = {
        "objective": objective,
        "loss": "multinomial_cross_entropy",
        "n_class": 5,
        "shift_cost": shift_cost
    }

    path = get_path([dataset, var_to_str(model_cfg), optimizer], out_path=result_dir)
    f = os.path.join(path, f"best_weights.p")
    return [iterate.view(-1, n_class).detach() for iterate in pickle.load(open(f, "rb"))]



def get_group_error(column, optimizer, objective):
    iterates = get_iterates(objective, optimizer)

    df = metadata.copy()

    for i, w in enumerate(iterates):
        logits = X_test @ w
        df[f"error_{i}"] = (y_test != torch.argmax(logits, dim=1)).int()

    cols = [column] + [f"error_{i}" for i in range(len(iterates))]

    # returns (T, G) numpy array where T is the number of iterates and G is the number of groups.
    return df[cols].groupby([column]).mean().to_numpy().T


if __name__ == "__main__":
    
    dataset = "amazon"
    loss = "multinomial_cross_entropy"
    n_class = 5
    lp_reg = 1.0
    shift_cost = 0.0

    result_dir = "../results"
    objective = "extremile"
    column = 4
    optimizer = "sorel"


    X = np.load(f"../data/{dataset}/X_test.npy")
    y = np.load(f"../data/{dataset}/y_test.npy")

    X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5, random_state=42)
    z_test = torch.tensor(np.load("../data/amazon/z_test.npy"))
    
    row_index = []
    for i in range(X_test.shape[0]):
        # have duplicate rows in X_test
        row_index_ = np.where((X == X_test[i,:]).all(axis=1))[0]
        row_index.append(row_index_[0])

    row_index = np.array(row_index).flatten()
    print(len(row_index))

    z_test = z_test[row_index]
    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)



    metadata = pd.DataFrame(z_test).drop(columns=[0, 1, 5])
    metadata.info()

    metadata.head()

    group_error = get_group_error(column, optimizer, objective)
    yy = np.max(group_error, axis=1)

    print(f"{optimizer} {objective} mean: {yy.mean():.4f}")
    print(f"{optimizer} {objective} std: {yy.std()}")



