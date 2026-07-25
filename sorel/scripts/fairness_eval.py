import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split


sys.path.append("..")
from src.utils.io import get_path, var_to_str


from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from tqdm import tqdm


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_weights(dataset, objective, optimizer, shift_cost, result_dir="../results/"):
  
    loss = "squared_error" if dataset == "acsincome" else "binary_cross_entropy"
    model_cfg = {
        "objective": objective,
        "loss": loss,
        "n_class": None,
        "shift_cost": shift_cost
    }

    path = get_path([dataset, var_to_str(model_cfg), optimizer], out_path=result_dir)
    f = os.path.join(path, f"best_weights.p")
    return [iterate.detach() for iterate in pickle.load(open(f, "rb"))]

def get_dp(X_test, y_test, column, optimizer, objective, shift_cost, result_dir, metric="difference"):
    iterates = get_weights(dataset, objective, optimizer, shift_cost, result_dir)
    out = []
    acc = []
    for i, w in tqdm(enumerate(iterates)):
        y_pred = (X_test @ w >= 0).int()
        if metric == "difference":
            out.append(demographic_parity_difference(y_test, y_pred, sensitive_features=list(df[column])))
        elif metric == "ratio":
            out.append(demographic_parity_ratio(y_test, y_pred, sensitive_features=list(df[column])))
        acc.append(torch.sum(y_pred == y_test).item() / len(y_test))
    return out, acc

def get_ks_dist(y_pred, y_true):
    out = ks_2samp(y_pred, y_true, method="asymp")
    return out.statistic

def get_dist_groups(X_test, y_test, column, optimizer, objective, shift_cost, result_dir, dataset="acsincome"):
    iterates = get_weights(dataset, objective, optimizer, shift_cost, result_dir)
    groups = list(df[column].unique())

    dist_groups = []

    for i, w in enumerate(iterates):
        y_pred = (X_test @ w).numpy()

        y_groups = [y_pred[df[column] == group] for group in groups]

        dist_groups.append(np.array([get_ks_dist(y_pred, y_group) for y_group in y_groups]))


    return np.array(dist_groups)


if __name__ == "__main__":
    dataset = "acsincome" 
    protected_attrib = "SEX" # "SEX" or "RAC1P"
    shift_cost = 0.0
    result_dir = "../results"
    optimizer = "sorel"

    objectives = ["superquantile", "esrm"]

    X = np.load(f"../data/{dataset}/X_test.npy")
    y = np.load(f"../data/{dataset}/y_test.npy")

    X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

    row_index = []
    for i in range(X_test.shape[0]):
        # have duplicate rows in X_test
        row_index_ = np.where((X == X_test[i,:]).all(axis=1))[0]
        row_index.append(row_index_[0])

    row_index = np.array(row_index).flatten()


    df_all = pd.read_csv(f"../data/{dataset}/metadata_te.csv")
    df = df_all.iloc[row_index]
    print(df.shape)

    X_test = torch.tensor(X[row_index])
    y_test = torch.tensor(y[row_index]).double()


    if dataset == "acsincome":    
        column = protected_attrib
        metric = "difference"
        print(df.SEX.unique())
        print(df.RAC1P.unique())    
    else:
        raise NotImplementedError

    print(f"+++++++fairness evaluation: in {dataset} with protected attribute {protected_attrib}+++++++")
    for j, objective in enumerate(objectives):
        sp_mean = []
        sp_std = []
        print(f"++++++++++++++++++++in spectral risk: {objective}++++++++++++++++++++++++++")
       
            
        group_loss = get_dist_groups(X_test, y_test, column, optimizer, objective, shift_cost, result_dir)
        yy = np.max(group_loss, axis=1) 

        # average over random seeds
        sp_mean.append(np.array(yy).mean())
        sp_std.append(np.array(yy).std())


        if dataset == "acsincome":
            print(f"worst dist mean {optimizer}: {np.array(yy).mean():.4f}")
            print(f"worst dist std {optimizer}: {np.array(yy).std()}")
        else:
            raise NotImplementedError












