import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted
from numba import jit
import torch.nn.functional as F

class Optimizer:
    def __init__(self):
        pass

    def start_epoch(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_epoch(self):
        raise NotImplementedError

    def get_epoch_len(self):
        raise NotImplementedError

class PSGSRM(Optimizer):
    def __init__(
        self,
        objective,
        lrp=0.01,
        lrd=None,
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        shift_cost=1.0,
        penalty="l2",
        oracle_reg="grad",
    ):
        super(PSGSRM, self).__init__()
        self.objective = objective
        self.lrp = lrp
        self.lrd = 1.0 if lrd is None else lrd
        n, d = self.objective.n, self.objective.d
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
            self.grad_table = torch.zeros(n, objective.n_class * d, dtype=torch.float64)
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
            self.grad_table = torch.zeros(n, d, dtype=torch.float64)
        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)
        self.shift_cost = n * shift_cost
        self.penalty = penalty
        assert oracle_reg in ["prox", "grad"]
        self.oracle_reg = oracle_reg
        
        # Generate loss and gradient tables.
        self.losses = self.objective.get_indiv_loss(self.weights).detach()
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        self.rho = self.lam.clone()
        real_lp_reg = self.objective.lp_reg / n

        if self.oracle_reg == "grad":
            self.grad_table = self.objective.get_indiv_grad(self.weights) 
        else:
            self.grad_table = self.objective.get_indiv_grad(self.weights)
        self.running_subgrad = torch.matmul(self.grad_table.T, self.rho)

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = self.objective.n

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n
        real_lp_reg = self.objective.lp_reg / n

        # Compute gradient at current iterate.
        i = torch.tensor([self.rng_grad.randint(0, n)])
        x = self.objective.X[i]
        y = self.objective.y[i]
        loss = self.objective.loss(self.weights, x, y)
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        

        # Compute gradient at from table.
        g_old = self.grad_table[i].reshape(-1)

        v = n * self.lam[i] * g - n * self.rho[i] * g_old + self.running_subgrad

        # Update iterate.
        if self.oracle_reg == "prox":
            self.weights.copy_(
                1 / (self.lrp * real_lp_reg + 1) * (self.weights - self.lrp * v)
            )
        else:
            # Proximal update
            self.weights.copy_(F.softshrink(self.weights - self.lrp * v, lambd=real_lp_reg))
            

        # update dual weights
        self.losses[i] = loss.detach()
        cur_lam = self.lam[i]
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        rho_old = self.rho[i]
        self.rho[i] = cur_lam

        # update table
        self.grad_table[i] = g[None, :]
        self.running_subgrad += cur_lam * g - rho_old * g_old

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len

     
class SmoothedLSVRG(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=1.0,
        smoothing="l2",
        seed=25,
        length_epoch=None,
    ):
        super(SmoothedLSVRG, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        self.lr = lr
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
        self.spectrum = self.objective.sigmas
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smoothing = smoothing
        if length_epoch:
            self.length_epoch = length_epoch
        else:
            self.length_epoch = int(nb_passes * n)
        self.nb_checkpoints = 0
        self.step_no = 0

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n

        # start epoch
        if self.step_no % n == 0:
            losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
            sorted_losses, self.argsort = torch.sort(losses, stable=True)
            self.sigmas = get_smooth_weights_sorted(
                sorted_losses, self.spectrum, self.smooth_coef, self.smoothing
            )
            with torch.enable_grad():
                self.subgrad_checkpt = self.objective.get_batch_subgrad(self.weights, include_reg=False)
            self.weights_checkpt = torch.clone(self.weights)
            self.nb_checkpoints += 1

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
        else:
            i = torch.tensor([np.random.choice(n, p=self.sigmas)])
        x = self.objective.X[self.argsort[i]]
        y = self.objective.y[self.argsort[i]]

        # Compute gradient at current iterate.
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        g_checkpt = self.objective.get_indiv_grad(self.weights_checkpt, x, y).squeeze()

        if self.uniform:
            direction = n * self.sigmas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        if self.objective.lp_reg:
            direction += self.objective.lp_reg * self.weights / n

        self.weights.copy_(self.weights - self.lr * direction)
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch
    

@jit(nopython=True)
def bubble_sort(idx, sort, argsort):
    n = len(sort)

    # Bubble left.
    j = idx
    while j > 0 and sort[j] < sort[j - 1] - 1e-10:
        # Swap elements in sorted vector.
        temp = sort[j]
        sort[j] = sort[j - 1]
        sort[j - 1] = temp

        # Swap elements in "argsort" vector.
        temp = argsort[j]
        argsort[j] = argsort[j - 1]
        argsort[j - 1] = temp

        j -= 1

    # Bubble right.
    j = idx
    while j < n - 1 and sort[j] > sort[j + 1] + 1e-10:
        # Swap elements in sorted vector.
        temp = sort[j]
        sort[j] = sort[j + 1]
        sort[j + 1] = temp

        # Swap elements in "argsort" vector.
        temp = argsort[j]
        argsort[j] = argsort[j + 1]
        argsort[j + 1] = temp

        j += 1


class StochasticSubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01, batch_size=64, seed=25, epoch_len=None):
        super(StochasticSubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr
        self.batch_size = batch_size

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
        self.order = None
        self.iter = None
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0

    def step(self):
        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        self.weights.requires_grad = True
        g = self.objective.get_batch_subgrad(self.weights, idx=idx)
        self.weights.requires_grad = False
        self.weights = self.weights - self.lr * g
        self.iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len

class Prospect(Optimizer):
    def __init__(
        self,
        objective,
        lrp=0.01,
        lrd=None,
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        shift_cost=1.0,
        penalty="l2",
        oracle_reg="grad",
    ):
        super(Prospect, self).__init__()
        self.objective = objective
        self.lrp = lrp
        self.lrd = 1.0 if lrd is None else lrd
        n, d = self.objective.n, self.objective.d
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
            self.grad_table = torch.zeros(n, objective.n_class * d, dtype=torch.float64)
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
            self.grad_table = torch.zeros(n, d, dtype=torch.float64)
        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)
        self.shift_cost = n * shift_cost
        self.penalty = penalty
        assert oracle_reg in ["prox", "grad"]
        self.oracle_reg = oracle_reg

        # Generate loss and gradient tables.
        self.losses = self.objective.get_indiv_loss(self.weights).detach()
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        self.rho = self.lam.clone()
        real_lp_reg = self.objective.lp_reg / n

        if self.oracle_reg == "grad":
            self.grad_table = self.objective.get_indiv_grad(self.weights) + real_lp_reg * self.weights[None, :]
        else:
            self.grad_table = self.objective.get_indiv_grad(self.weights)
        self.running_subgrad = torch.matmul(self.grad_table.T, self.rho)

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = self.objective.n

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n
        real_lp_reg = self.objective.lp_reg / n

        # Compute gradient at current iterate.
        i = torch.tensor([self.rng_grad.randint(0, n)])
        x = self.objective.X[i]
        y = self.objective.y[i]
        loss = self.objective.loss(self.weights, x, y)
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        if self.oracle_reg == "grad":
            g += real_lp_reg * self.weights

        # Compute gradient at from table.
        g_old = self.grad_table[i].reshape(-1)

        v = n * self.lam[i] * g - n * self.rho[i] * g_old + self.running_subgrad

        # Update iterate.
        if self.oracle_reg == "prox":
            self.weights.copy_(
                1 / (self.lrp * real_lp_reg + 1) * (self.weights - self.lrp * v)
            )
        else:
            self.weights.copy_(self.weights - self.lrp * v)

        # update dual weights
        # self.losses[i] = self.lrd*loss + (1-self.lrd)*self.losses[i]
        self.losses[i] = loss.detach()
        cur_lam = self.lam[i]
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        rho_old = self.rho[i]
        self.rho[i] = cur_lam

        # update table
        self.grad_table[i] = g[None, :]
        self.running_subgrad += cur_lam * g - rho_old * g_old

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len    