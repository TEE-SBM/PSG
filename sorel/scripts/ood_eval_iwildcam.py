
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import pickle
import numpy as np
import pandas as pd

sys.path.append("..")
from src.utils.io import var_to_str, get_path
from src.utils.data import load_dataset





def get_iterates(objective, optimizer, seed=0, dataset="iwildcam"):
    model_cfg = {
        "objective": objective,
        "loss": "multinomial_cross_entropy",
        "n_class": 60,
        "shift_cost": shift_cost
    }

    path = get_path([dataset, var_to_str(model_cfg), optimizer], out_path=result_dir)
    f = os.path.join(path, f"best_weights.p")
    return [iterate.view(-1, n_class).detach() for iterate in pickle.load(open(f, "rb"))]




def get_group_error(optimizer, objective):
    iterates = get_iterates(objective, optimizer)

    df = metadata.copy()

    for i, w in enumerate(iterates):
        logits = X_test @ w
        df[f"error_{i}"] = (y_test != torch.argmax(logits, dim=1)).int()

    # returns (T, G) numpy array where T is the number of iterates and G is the number of groups.
    return df.groupby([0]).mean().to_numpy().T




if __name__ == "__main__":
    dataset = "iwildcam"
    loss = "multinomial_cross_entropy"
    n_class = 60
    objective = "extremile"
    optimizer = "sorel"

    result_dir = "../results"
    lp_reg = 1.0
    shift_cost = 0.0

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset, data_path="../data/")

    metadata = pd.DataFrame(y_test)
    metadata.head()



    group_error = get_group_error(optimizer, objective)
    median_group_error = np.quantile(group_error, 0.5, axis=1)
    average_median_group_error = np.mean(median_group_error)
    std_median_group_error = np.std(median_group_error)


    print(f"Optimizer: {optimizer}, Objective: {objective}, \
          Average Median Group Error: {average_median_group_error:.4f}, \
            Std Median Group Error: {std_median_group_error}")


