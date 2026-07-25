import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted
from src.optim.baselines import Optimizer
from numba import jit
import warnings


class SorelBatch(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=0,
        smoothing="l2",
        seed=25,
        length_epoch=None,
        weights=None,
        lrdcon = None,
        xlrcon = None,
        batch_size=64,
        change_weights = False,
    ):
        super(SorelBatch, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        # primal learning rate
        self.lr = lr
        self.batch_size = batch_size
        if weights is None:
            if objective.n_class:
                self.weights = torch.zeros(
                    objective.n_class * d,
                    requires_grad=True,
                    dtype=torch.float64,
                )
            else:
                if change_weights is False:
                    self.weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
                else:
                    self.weights = torch.ones(d, requires_grad=True, dtype=torch.float64)
        else:
            self.weights = weights
        self.spectrum = self.objective.sigmas
        torch.manual_seed(seed)
        self.uniform = uniform
        self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smoothing = smoothing
        if length_epoch:
            self.length_epoch = min(length_epoch, self.objective.n // self.batch_size)
        else:
            self.length_epoch = self.objective.n // self.batch_size

        self.nb_checkpoints = 0
        self.step_no = 0
        if lrdcon is None:
            self.lrdcon = 1e-1
        else:
            self.lrdcon = lrdcon
        if xlrcon is None:
            self.xlrcon = 20
        else:
            self.xlrcon = xlrcon
        self.lrd = self.lrdcon/self.objective.n
        self.xlr = self.xlrcon * self.objective.n
        # momentum parameter
        self.theta = 0
        self.weights_prev = self.weights
        self.dual_iter = 0

        losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
        self.sigmas = self.spectrum[torch.argsort(torch.argsort(losses))]
        self.batch_size = batch_size
        self.losses_prev = losses

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0
        self.dual_update()
        with torch.enable_grad():
            loss_gradient = self.objective.get_indiv_grad(self.weights)
        self.subgrad_checkpt = torch.matmul(self.sigmas, loss_gradient)
        # with torch.enable_grad():
        #     self.subgrad_checkpt = self.objective.get_batch_subgrad(self.weights, include_reg=False)
        self.weights_checkpt = torch.clone(self.weights)
        self.nb_checkpoints += 1

    def dual_update(self):
        losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
        v = losses + self.theta*(losses-self.losses_prev)
        self.losses_prev = losses

        self.sigmas = proj_perm(v, self.spectrum, 1/self.lrd, self.sigmas)
        self.step_no = 0
        self.weights_prev = torch.clone(self.weights)
        self.dual_iter += 1
        self.theta = self.dual_iter/(self.dual_iter + 1)
        if self.dual_iter <= 400:
            self.lrd = self.lrdcon*(self.dual_iter + 1)/self.objective.n
        self.xlr = self.xlrcon * self.objective.n /(self.dual_iter + 1)


    @torch.no_grad()
    def step(self):
        n = self.objective.n

        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        self.weights.requires_grad = True
        # g = self.objective.get_batch_subgrad(self.weights, idx=idx)
        g = self.objective.get_indiv_grad_idx(self.weights, idx)
        self.weights.requires_grad = False
        self.iter += 1

        # Compute gradient at current iterate.
        g_checkpt = self.objective.get_indiv_grad_idx(self.weights_checkpt, idx=idx)

        direction = n * torch.matmul(self.sigmas[idx], (g-g_checkpt)) / self.batch_size + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction += self.objective.l2_reg * self.weights / n + 1/self.xlr*(self.weights-self.weights_prev)
        self.weights.copy_(self.weights - self.lr * direction)
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch
    


def proj_perm(losses, spectrum, smooth_coef, y_prev):
    if smooth_coef < 1e-16:
        _, perm = torch.sort(losses, stable=True)
        return spectrum[torch.argsort(perm)]
    n = len(losses)
    scaled_losses = losses / smooth_coef
    y_intern_scaled = scaled_losses + y_prev
    perm = torch.argsort(y_intern_scaled)
    y_intern_sorted = y_intern_scaled[perm]

    primal_sol = l2_centered_isotonic_regression_new(
        y_intern_sorted.numpy(), spectrum.numpy())
    
    inv_perm = torch.argsort(perm)
    primal_sol = primal_sol[inv_perm]
    smooth_weights = y_intern_scaled - primal_sol 
    return smooth_weights

def l2_centered_isotonic_regression_new(losses, spectrum):
    n = len(losses)
    means = [losses[0] - spectrum[0]]
    counts = [1]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] - spectrum[i])
        counts.append(1)
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_count, prev_end_point = (
                means.pop(),
                counts.pop(),
                end_points.pop(),
            )
            means[-1] = (counts[-1] * means[-1] + prev_count * prev_mean) / (
                counts[-1] + prev_count
            )
            counts[-1] = counts[-1] + prev_count
            end_points[-1] = prev_end_point
            
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol