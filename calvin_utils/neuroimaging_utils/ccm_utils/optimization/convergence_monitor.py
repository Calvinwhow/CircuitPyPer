import numpy as np

class ConvergenceMonitor:
    """
    Stop-criteria aggregator for iterative optimisation.

    Triggers ‘converged = True’ when **any** of the following is satisfied
    (after an initial warm-up period):

    • iteration limit                       –  iterations ≥ max_iterations  
    • small gradient norm                   –  ‖g‖₂   < grad_tol  
    • small parameter update                –  ‖ΔW‖₂  < param_tol  
    • small loss change since last step     –  |ΔL|   < loss_tol  
    • plateau of gradient norm              –  max(‖g‖) − min(‖g‖) < grad_tol over plateau_window  
    • plateau of loss                       –  max(L) − min(L)    < loss_tol  over plateau_window
    """
    def __init__(self,
                 tol: float        = 1e-5,          # (kept for backwards-compat; unused)
                 max_i: int        = 500,
                 grad_tol: float   = 1e-6,
                 param_tol: float  = 1e-6,
                 loss_tol: float   = 1e-6,
                 plateau_window: int = 20,
                 warmup: int       = 25):
        self.max_iterations = max_i
        self.grad_tol       = grad_tol
        self.param_tol      = param_tol
        self.loss_tol       = loss_tol
        self.plateau_window = plateau_window
        self.warmup         = warmup

        self.iterations     = 0
        self.prev_weights   = None
        self.prev_loss      = None
        self.grad_hist      = []      # ‖g‖ history
        self.loss_hist      = []      # loss history

    # ------------------------------------------------------------------
    def _plateau(self, history: list, tol: float) -> bool:
        if len(history) < self.plateau_window:
            return False
        recent = np.asarray(history[-self.plateau_window:])
        return (recent.max() - recent.min()) < tol

    # ------------------------------------------------------------------
    def check_convergence(self,
                          weights:  np.ndarray,
                          gradient: np.ndarray = None,
                          loss:     float = None) -> bool:
        """
        Update internal state and return *True* as soon as convergence is met.
        """
        self.iterations += 1

        # ----- compute diagnostics BEFORE overwriting prev_* -----
        grad_norm   = np.linalg.norm(gradient) if gradient is not None else np.inf
        param_delta = (np.linalg.norm(weights - self.prev_weights)
                       if self.prev_weights is not None else np.inf)
        loss_delta  = (abs(loss - self.prev_loss)
                       if (loss is not None and self.prev_loss is not None) else np.inf)

        # ----- record current step -----
        self.prev_weights = np.copy(weights)
        self.prev_loss    = loss
        self.grad_hist.append(grad_norm)
        self.loss_hist.append(loss)

        # ----- early-stage guard -----
        if self.iterations <= self.warmup:
            return False

        # ----- stopping conditions -----
        if self.iterations >= self.max_iterations:
            print(f"[Convergence] Reached max iterations ({self.max_iterations}).")
            return True

        if grad_norm < self.grad_tol:
            print(f"[Convergence] Gradient norm {grad_norm:.3e} < grad_tol.")
            return True

        if param_delta < self.param_tol:
            print(f"[Convergence] Parameter change {param_delta:.3e} < param_tol.")
            return True

        if loss_delta < self.loss_tol:
            print(f"[Convergence] Loss change {loss_delta:.3e} < loss_tol.")
            return True

        if self._plateau(self.grad_hist, self.grad_tol):
            print(f"[Convergence] Gradient norm plateaued (< grad_tol) over "
                  f"last {self.plateau_window} steps.")
            return True

        if self._plateau(self.loss_hist, self.loss_tol):
            print(f"[Convergence] Loss plateaued (< loss_tol) over "
                  f"last {self.plateau_window} steps.")
            return True

        return False
