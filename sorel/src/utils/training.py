import pandas as pd
import time
from tqdm import tqdm
import torch
import pickle
import os
import datetime

from src.optim.SOREL import Sorel



from src.optim.objectives import (
    Objective,
    get_extremile_weights,
    get_superquantile_weights,
    get_esrm_weights,
    get_erm_weights,
)
from src.utils.io import save_results, load_results, var_to_str, get_path
from src.utils.data import load_dataset

SUCCESS_CODE = 0
FAIL_CODE = -1

class OptimizationError(RuntimeError):
    pass


def train_model(optimizer, val_objective, test_objective, n_epochs, metric_type):
    epoch_len = optimizer.get_epoch_len()
    metrics = [compute_metrics(-1, optimizer, val_objective, test_objective, 0.0, metric_type)]
    init_loss = metrics[0]["train_loss"]

    for epoch in tqdm(range(n_epochs)):
        tic = time.time()
        optimizer.start_epoch()
        for _ in range(epoch_len):
            optimizer.step()
        optimizer.end_epoch()
        toc = time.time()

        # Logging.
        metrics.append(compute_metrics(epoch, optimizer, val_objective, test_objective, toc - tic, metric_type))
        if metrics[-1]["train_loss"] >= 1.5 * init_loss:
            raise OptimizationError(
                f"train loss 50% greater than inital loss! (epoch {epoch})"
            )

    result = {
        "weights": optimizer.weights,
        "metrics": pd.DataFrame(metrics),
    }
    return result


def get_optimizer(optim_cfg, objective, seed, device="cpu"):
    name, lr, epoch_len, shift_cost = (
        optim_cfg["optimizer"],
        optim_cfg["lr"],
        optim_cfg["epoch_len"],
        optim_cfg["shift_cost"],
    )

    lrd = 0.5 if "lrd" not in optim_cfg.keys() else optim_cfg["lrd"]
    penalty = "l2"


    if name == "sorel":
        return Sorel(
            objective,
            lr=lr,
            smooth_coef=shift_cost,
            smoothing=penalty,
            seed=seed,
            length_epoch=epoch_len,
        )        
    else:
        raise ValueError("Unreocgnized optimizer!")


def get_objective(model_cfg, X, y, dataset=None, autodiff=True):
    name, lp_reg, reg_type, loss, n_class, shift_cost = (
        model_cfg["objective"],
        model_cfg["lp_reg"],
        model_cfg["reg_type"],
        model_cfg["loss"],
        model_cfg["n_class"],
        model_cfg["shift_cost"],
    )
    if name == "erm":
        weight_function = lambda n: get_erm_weights(n)
    elif name == "extremile":
        weight_function = lambda n: get_extremile_weights(n, 2.0)
    elif name == "superquantile":
        weight_function = lambda n: get_superquantile_weights(n, 0.5)
    elif name == "esrm":
        weight_function = lambda n: get_esrm_weights(n, 1.0)
    elif name == "extremile_lite":
        weight_function = lambda n: get_extremile_weights(n, 1.5)
    elif name == "superquantile_lite":
        weight_function = lambda n: get_superquantile_weights(n, 0.25)
    elif name == "esrm_lite":
        weight_function = lambda n: get_esrm_weights(n, 0.5)
    elif name == "extremile_hard":
        weight_function = lambda n: get_extremile_weights(n, 2.5)
    elif name == "superquantile_hard":
        weight_function = lambda n: get_superquantile_weights(n, 0.75)
    elif name == "esrm_hard":
        weight_function = lambda n: get_esrm_weights(n, 2.0)

    return Objective(
        X,
        y,
        weight_function,
        lp_reg=lp_reg,
        reg_type=reg_type,
        loss=loss,
        n_class=n_class,
        risk_name=name,
        dataset=dataset,
        shift_cost=shift_cost,
        penalty="l2",
        autodiff=autodiff,
    )


def compute_metrics(epoch, optimizer, val_objective, test_objective, elapsed, metric_type="general"):  
    if metric_type == "general":
        return {
        "epoch": epoch,
        "train_loss": optimizer.objective.get_batch_loss(optimizer.weights).item(),
        "train_loss_unreg": optimizer.objective.get_batch_loss(optimizer.weights, include_reg=False).item(),
        "val_loss": val_objective.get_batch_general_loss(optimizer.weights).item(),
        "test_loss": test_objective.get_batch_general_loss(optimizer.weights).item(),
        "elapsed": elapsed,
        }
    elif metric_type == "spectral":
        return {
        "epoch": epoch,
        "train_loss": optimizer.objective.get_batch_loss(optimizer.weights).item(),
        "train_loss_unreg": optimizer.objective.get_batch_loss(optimizer.weights, include_reg=False).item(),
        "val_loss": val_objective.get_batch_spectral_loss(optimizer.weights).item(),
        "test_loss": test_objective.get_batch_spectral_loss(optimizer.weights).item(),
        "elapsed": elapsed,
        }
    else:
        raise ValueError("Unrecognized metric type!")

def compute_training_curve(
    dataset,
    model_cfg,
    optim_cfg,
    seed,
    n_epochs,
    metric_type,
    out_path="results/",
    data_path="data/"
):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset, data_path=data_path)

    if model_cfg["loss"] == "multinomial_cross_entropy":
        model_cfg["n_class"] = len(torch.unique(y_train))

    # check if result exists
    if (
        result_exists(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
    ):
        print("*** Result exists ***")
        print("*********************")
        print(f"dataset: {dataset}")
        print(f"model_cfg: {model_cfg}")
        print(f"optim_cfg: {optim_cfg}")
        print(f"seed: {seed}")
        print("*********************")
        exit_code = SUCCESS_CODE
    else:
        train_objective = get_objective(model_cfg, X_train, y_train, dataset=dataset, autodiff=False)
        val_objective = get_objective(model_cfg, X_val, y_val, dataset=dataset, autodiff=False)
        test_objective = get_objective(model_cfg, X_test, y_test, dataset=dataset, autodiff=False)
        
        optimizer = get_optimizer(optim_cfg, train_objective, seed)
        try:
            result = train_model(optimizer, val_objective, test_objective, n_epochs, metric_type)
            exit_code = SUCCESS_CODE
        except OptimizationError as e:
            result = FAIL_CODE
            exit_code = FAIL_CODE
        save_results(result, dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        return exit_code


def result_exists(dataset, model_cfg, optim_cfg, seed, out_path="results"):
    key_ = ["objective", "shift_cost", "loss", "n_class"]
    model_cfg_ = {k: model_cfg[k] for k in key_}
    path = "/".join([out_path, dataset, var_to_str(model_cfg_), var_to_str(optim_cfg)])
    f = os.path.join(path, f"seed_{seed}.p")
    return os.path.exists(f)


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def compute_average_val_loss(
    dataset, model_cfg, optim_cfg, seeds, out_path="results/"
):
    total = 0.0
    for seed in seeds:
        results = load_results(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            return [torch.inf]
        total += torch.tensor(results["metrics"]["val_loss"])
    return total / len(seeds)

def find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds, out_path="results/"):
    # Compute optimal hyperparameters by lowest average final train loss.
    best_loss = torch.inf
    best_traj = None
    best_cfg = None
    for optim_cfg in optim_cfgs:
        avg_val_loss = compute_average_val_loss(
            dataset, model_cfg, optim_cfg, seeds, out_path=out_path
        )
        # if len(avg_train_loss) > 1 and torch.trapezoid(avg_train_loss) < best_loss:
        if len(avg_val_loss) > 1 and torch.mean(avg_val_loss[-10:]) < best_loss:
            best_loss = torch.mean(avg_val_loss[-10:])
            best_traj = avg_val_loss
            best_cfg = optim_cfg

    # Collect results for best configuration.
    df = pd.DataFrame(
        {
            "epoch": [i for i in range(len(best_traj))],
            "average_val_loss": [val.item() for val in best_traj],
        }
    )

    key_ = ["objective", "shift_cost", "loss", "n_class"]
    model_cfg_ = {k: model_cfg[k] for k in key_}
    path = get_path([dataset, var_to_str(model_cfg_), optim_cfgs[0]["optimizer"]], out_path=out_path)

    weights = []
    for seed in seeds:
        results = load_results(dataset, model_cfg, best_cfg, seed, out_path=out_path)
        df[f"seed_{seed}_train"] = results["metrics"]["train_loss"]
        df[f"seed_{seed}_val"] = results["metrics"]["val_loss"]
        df[f"seed_{seed}_test"] = results["metrics"]["test_loss"]
        if "nb_checkpoints" in results.keys():
            nb_checkpoints = results["nb_checkpoints"]
            pickle.dump(
                nb_checkpoints, open(os.path.join(path, "nb_checkpoints.p"), "wb")
            )

        weights.append(results["weights"])

    print("Saving results to location:")
    print(path)
    print(best_cfg)
    pickle.dump(best_cfg, open(os.path.join(path, "best_cfg.p"), "wb"))
    pickle.dump(weights, open(os.path.join(path, "best_weights.p"), "wb"))
    pickle.dump(df, open(os.path.join(path, "best_traj.p"), "wb"))
