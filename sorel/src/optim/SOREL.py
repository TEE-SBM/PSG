import torch
import numpy as np
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

class Sorel(Optimizer):
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
        average = False,
        change_weights = False,
    ):
        super(Sorel, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        # average_epoch update
        self.average = average
        # primal learning rate
        self.lr = lr
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

        self.average_weitghts = self.weights if self.average else None

        losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
        self.sigmas = self.spectrum[torch.argsort(torch.argsort(losses))]
        self.sigmas_prev = torch.clone(self.sigmas) # for debug

    def start_epoch(self):
        self.dual_update()

    def dual_update(self):
        self.sigmas_prev = torch.clone(self.sigmas) # for debug
        losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
        losses_prev = self.objective.get_indiv_loss(self.weights_prev, with_grad=False)
        v = losses + self.theta*(losses-losses_prev)
        self.sigmas = proj_perm(v, self.spectrum, 1/self.lrd,self.sigmas)

        self.step_no = 0
        self.weights_prev = torch.clone(self.weights)
        self.dual_iter += 1
        self.theta = self.dual_iter/(self.dual_iter + 1)
        self.lrd = self.lrdcon*(self.dual_iter + 1)/(self.objective.n)**1
        self.xlr = self.xlrcon * self.objective.n /(self.dual_iter + 1)
        
    @torch.no_grad()
    def step(self):
        n = self.objective.n
        real_lp_reg = self.objective.lp_reg / n
        if self.step_no % (np.floor(n)) == 0:
            with torch.enable_grad():
                loss_gradient = self.objective.get_indiv_grad(self.weights)
            self.subgrad_checkpt = torch.matmul(self.sigmas, loss_gradient)
            self.weights_checkpt = torch.clone(self.weights)
            self.nb_checkpoints += 1

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
        else:
            i = torch.tensor([np.random.choice(n, p=self.sigmas)])
        x = self.objective.X[i]
        y = self.objective.y[i]

        # Compute gradient at current iterate.
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        g_checkpt = self.objective.get_indiv_grad(self.weights_checkpt, x, y).squeeze()

        if self.uniform:
            direction = n * self.sigmas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        
        if self.objective.lp_reg and self.objective.reg_type == "l2":
            direction += real_lp_reg * self.weights + 1/self.xlr*(self.weights-self.weights_prev)
            self.weights.copy_(self.weights - self.lr * direction)
        elif self.objective.lp_reg and self.objective.reg_type == "l1":
            curr_weights = self.weights - self.lr * direction
            weights_ = self.xlr / (self.lr + self.xlr) * curr_weights \
                            + self.lr / (self.lr + self.xlr) * self.weights_prev
            lambd_ = real_lp_reg * self.lr * self.xlr / (self.lr + self.xlr)
            self.weights.copy_(F.softshrink(weights_, lambd=lambd_))
        else:
            raise NotImplementedError("Without regularization not implemented")
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch
    

def proj_perm(losses, spectrum, smooth_coef, y_prev):
    if smooth_coef < 1e-16:
        _, perm = torch.sort(losses, stable=True)
        return spectrum[torch.argsort(perm)]
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


