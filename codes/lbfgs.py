"""
Run L-BFGS optimizer to get optimal value of spectral risk for a given dataset and regularizer.
Used to compute suboptimality of the optimizers assessed.
"""

import os
import sys
import numpy as np
import torch
from scipy.optimize import minimize
import pickle
import argparse

sys.path.append(".")
from src.utils.data import load_dataset
from src.utils.training import get_objective, OptimizationError
from src.utils.io import var_to_str, get_path


def run_lbfgs(args, L1_REG, SHIFT_COST):
    dataset = args.dataset
    if dataset in ["yacht", "energy", "concrete", "kin8nm", "power", "acsincome"]:
        loss = "squared_error"
        n_class = None
    elif dataset == "iwildcam":
        loss = "multinomial_cross_entropy"
        n_class = 60
    elif dataset == "amazon":
        loss = "multinomial_cross_entropy"
        n_class = 5
    elif dataset == "diabetes":
        loss = "binary_cross_entropy"

    model_cfg = {
        "objective": args.objective,
        "l1_reg": L1_REG,
        "shift_cost": SHIFT_COST,
        "loss": loss,
        "n_class": n_class
    }

    X_train, y_train, X_val, y_val = load_dataset(dataset)
    objective = get_objective(model_cfg, X_train, y_train)


    # Define function and Jacobian oracles.
    def fun(w):
        return objective.get_batch_loss(torch.tensor(w, dtype=torch.float64)).item()


    def jac(w):
        return (
            objective.get_batch_subgrad(
                torch.tensor(w, dtype=torch.float64, requires_grad=True)
            )
            .detach()
            .numpy()
        )


    # Run optimizer.
    init = np.zeros((objective.d,), dtype=np.float64)
    if model_cfg["n_class"]:
        init = np.zeros((model_cfg["n_class"] * objective.d,), dtype=np.float64)
    else:
        init = np.zeros((objective.d,), dtype=np.float64)
    output = minimize(fun, init, method="L-BFGS-B", jac=jac)
    if output.success:
        path = get_path([dataset, var_to_str(model_cfg)])
        f = os.path.join(path, "lbfgs_min_loss.p")
        pickle.dump(output.fun, open(f, "wb"))
    else:
        raise OptimizationError(output.message)
    
    
L1_REG = 1.0
SHIFT_COST = 1.0

# Create parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    choices=[
        "yacht",
        "energy",
        "simulated",
        "concrete",
        "iwildcam",
        "kin8nm",
        "power",
        "acsincome",
        "diabetes",
        "amazon",
    ],
)
parser.add_argument(
    "--objective",
    type=str,
    required=False,
    choices=[
        "extremile",
        "superquantile",
        "esrm",
        "erm",
        "extremile_lite",
        "superquantile_lite",
        "esrm_lite",
        "extremile_hard",
        "superquantile_hard",
        "esrm_hard",
    ],
)
args = parser.parse_args()
datasets = ["yacht", "energy", "concrete", "kin8nm", "power"]
objectives = ["extremile", "superquantile", "esrm"]

for dataset in datasets:
    for objective in objectives:
        args.dataset = dataset
        args.objective = objective
        run_lbfgs(args, L1_REG, SHIFT_COST)
        


