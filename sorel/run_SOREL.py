import time
import os
import psutil


import datetime
from joblib import Parallel, delayed
import sys
import argparse
# Create parser.
sys.path.append(".")
from src.utils.training import (
    compute_training_curve,
    format_time,
    find_best_optim_cfg,
    FAIL_CODE,
)
from src.utils.io import dict_to_list
os.environ["OMP_NUM_THREADS"] = "1"


def run(args, Lp_REG, SHIFT_COST, LRS, SEEDS):
    dataset = args.dataset
    if dataset in ["acsincome"]:
        loss = "squared_error"
        n_class = None
    elif dataset == "iwildcam":
        loss = "multinomial_cross_entropy"
        n_class = 60
    elif dataset == "amazon":
        loss = "multinomial_cross_entropy"
        n_class = 5


    model_cfg = {
        "objective": args.objective,
        "lp_reg": 1.0, # placeholder
        "reg_type": args.reg_type,
        "shift_cost": SHIFT_COST,
        "loss": loss,
        "n_class": n_class
    }


    lrs = LRS
    optim_cfg = {
        "optimizer": args.optimizer,
        "lr": lrs,
        "epoch_len": args.epoch_len,
        "shift_cost": SHIFT_COST,
        "lp_reg": Lp_REG, # placeholder
    }
    
    n_epochs = args.n_epochs
    metric_type = args.metric_type
    
    parallel = bool(args.parallel)

    optim_cfgs = dict_to_list(optim_cfg)

    config = {
        "dataset": dataset,
        "model_cfg": model_cfg,
        "optim_cfg": optim_cfg,
        "parallel": parallel,
        "seeds": SEEDS,
        "n_epochs": n_epochs,
        "epoch_len": args.epoch_len,
    }

    # Display.
    print("-----------------------------------------------------------------")
    for key in config:
        print(f"{key}:" + " " * (16 - len(key)), config[key])
    print(f"Start:" + " " * 11, {str(datetime.datetime.now())})
    print("-----------------------------------------------------------------")


    # Run optimization.
    def worker(optim):
        name, lr, lp_reg = optim["optimizer"], optim["lr"], optim["lp_reg"]
        print(f'running with optimizer {name} with lr={lr}, lp_reg={lp_reg}, reg_type={model_cfg["reg_type"]}...')
        model_cfg["lp_reg"] = lp_reg
        diverged = False
        for seed in SEEDS:
            code = compute_training_curve(
                dataset,
                model_cfg,
                optim,
                seed,
                n_epochs,
                metric_type,
            )
            if code == FAIL_CODE:
                diverged = True
        if diverged:
            print(f"Optimizer '{name}' diverged at learning rate {lr}, regularization {lp_reg}!")


    tic = time.time()
    if parallel:
        Parallel(n_jobs=args.n_jobs)(delayed(worker)(optim) for optim in optim_cfgs)
    else:
        for optim in optim_cfgs:
            worker(optim)
    toc = time.time()
    print(f"Time:         {format_time(toc-tic)}.")

    # Save best configuration.
    find_best_optim_cfg(dataset, model_cfg, optim_cfgs, SEEDS)
    

def params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="acsincome",
        choices=[
            "iwildcam",
            "acsincome",
            "amazon",
        ],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="extremile",
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
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",   
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=64,
    )
    
    parser.add_argument(
        "--reg_type",
        type=str,
        default="l1",
    )
    parser.add_argument(
        "--epoch_len",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--use_hyperparam",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--metric_type",
        type=str,
        default=None,
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":

    args = params()
    
    physical_cores = psutil.cpu_count(logical=False)
    args.n_jobs = physical_cores
    
    SHIFT_COST = 0.0
    LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]
    SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    METRIC_TYPES = "general"
    
    args.metric_type = METRIC_TYPES
    optimizer = "sorel"
    
    # For OOD classification
    # datasets = ["amazon", "iwildcam"] 
    # objectives = ["extremile"] 
    

    # For fairness evaluation
    datasets = ["acsincome"]
    objectives = ["superquantile", "esrm"]

    for dataset in datasets:
        if dataset in ["amazon"]:
            args.n_epochs = 32
        else:
            args.n_epochs = 64
        for objective in objectives:
                args.reg_type = "l1"
                Lp_REG = [0.01, 0.1, 1.0, 10, 100]
                args.dataset = dataset
                args.objective = objective
                args.optimizer = optimizer
                run(args, Lp_REG, SHIFT_COST, LRS, SEEDS)

    print("Finished training.")

