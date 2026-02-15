import numpy as np
from scipy.stats import spearmanr
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.ccm_utils.optimization.adam import AdamOptimizer
from calvin_utils.ccm_utils.optimization.convergence_monitor import ConvergenceMonitor
from tqdm import trange

class WeightOptimizer:
    """
    Gradient-descent optimiser over the weight vector W that combines
    multiple correlation maps into a single ‘convergent map’.
    All data I/O and similarity maths live on the parent NiftiOptimizer.
    """

    # -----------------------  construction  ---------------------------
    def __init__(self, parent, lr: float = 0.1, max_iters: int = 500):
        self.nifti      = parent                     # ↳ delegate heavy work
        self.W          = parent.W.copy()            # working copy
        self.best_loss  = -np.inf
        self.best_W     = self.W.copy()

        self.adam               = AdamOptimizer(self.W, lr=lr)
        self.convergence_monitor = ConvergenceMonitor(max_i=max_iters)

        self.h          = 1e-3                       # finite-diff ε
        self.iter_maps  = []                         # optional trace
        self.converged  = False

    # -------------------  helper / utility funcs  ---------------------
    def _tanh_normalize(self, W):        # keep weights bounded ±1 and Σ|w| = 1
        W = np.tanh(W)
        return W / np.sum(np.abs(W))

    def _clip(self, g, lo=-.5, hi=.5):    # gradient clipping
        return np.clip(g, lo, hi)

    # -----------------------  core maths  -----------------------------
    def _rho_array(self, avg_map):
        """Spearman ρ between similarity(X_i, AVG) and outcome for each dataset"""
        rhos = np.zeros(len(self.nifti.data_loader.dataset_names_list))
        for idx, k in enumerate(self.nifti.data_loader.dataset_names_list):

            if self.nifti.datasets is None:               # lazy-load
                dset = self.nifti.data_loader.load_dataset(k)
                X    = CorrelationCalculator._check_for_nans(dset['niftis'],    nanpolicy='remove')
                y    = CorrelationCalculator._check_for_nans(dset['indep_var'], nanpolicy='remove')
            else:                                          # pre-loaded
                X = self.nifti.datasets[k]['niftis']
                y = self.nifti.datasets[k]['indep_var']

            sim = self.nifti._calculate_similarity(X, avg_map)
            rho, _ = spearmanr(sim, y)
            rhos[idx] = rho
        return rhos

    @staticmethod
    def _target(rhos):
        '''Root Mean Squared Error equivalent with Rho'''
        return np.sqrt(np.mean(np.square(rhos)))

    def _penalty_all(self, thr=0.33, scale=1000):
        return np.sum(1e-6 / (thr - np.abs(self.W))) / scale

    def _penalty_each(self, thr=0.00, k=0.005):
        mask = np.abs(self.W) > thr
        return np.sum(np.abs(self.W) * mask * k)

    # ---------------------  loss + gradient  --------------------------
    def _loss(self, avg_map):
        rho_arr = self._rho_array(avg_map)
        t       = self._target(rho_arr)
        return t # penalties disabled as no need for boundary conditions

    def _forward_diff_grad(self, base_loss):
        '''Forward partial difference quotient to estimate gradient (direction of steepest ascent w.r.t. loss)'''
        g   = np.zeros_like(self.W)
        eye = np.eye(self.W.size).reshape(self.W.size, *self.W.shape) * self.h

        for i, dW in enumerate(eye):
            loss_fwd = self._loss(self.nifti._converge_maps(W=self.W + dW))
            g.flat[i] = (loss_fwd - base_loss) / self.h
        return self._clip(g).reshape(self.W.shape)

    # -----------------------  main routine  ---------------------------
    def optimise(self, store_iters=False, store_best=False):
        """
        Adam + FD gradient until convergence monitor halts.
        Returns final convergent map.
        """
        loss = 0
        for _ in trange(self.nifti.convergence_monitor.max_iterations, desc=f"Optimizing Weights, RMS Rho = {loss}"):
            if self.converged:
                break
            avg_map = self.nifti._converge_maps(W=self.W)
            loss    = self._loss(avg_map)

            grad = self._forward_diff_grad(loss)
            self.W = self.adam.step(grad)
            self.W = self._tanh_normalize(self.W)

            self.converged = self.convergence_monitor.check_convergence(
                weights=self.W, gradient=grad, loss=loss
            )

            if store_best and loss > self.best_loss:
                self.best_loss, self.best_W = loss, self.W.copy()
            if store_iters:
                self.iter_maps.append(avg_map)

        print("Selected weights:", self.W)
        return self.nifti._converge_maps(self.W)
    
    # ---------- second-stage blend optimiser ---------- #
    def blend_optimize(self, W_opt: np.ndarray, W_unw: np.ndarray, lam_delta: float = 0.05, lam_alpha: float = 0.01, n_grid: int = 200) -> tuple[float, np.ndarray]:
        """Search α∈[0,1] maximising J(alpha) by using switching function to blend optimal weights vs no weighting."""
                   # const
        def J(alpha: float) -> float:
            W = alpha * W_opt + (1 - alpha) * W_unw
            loss = self._loss(self.nifti._converge_maps(W))
            penalty = (alpha)**2 * (np.linalg.norm(W_opt - W_unw) ** 2) # Penalizes as alpha increases, which corresponds to larger deviations from W_unw
            return loss - penalty

        alphas = np.linspace(0, 1.0, n_grid)
        j_vals = [J(a) for a in alphas]
        k_best = int(np.argmax(j_vals))
        best_alpha = float(alphas[k_best])
        best_W     = best_alpha * W_opt + (1 - best_alpha) * W_unw
        print(f"Blend optimize: best_alpha={best_alpha}, best_W={best_W}")
        return best_alpha, best_W, self.nifti._converge_maps(W=best_W)