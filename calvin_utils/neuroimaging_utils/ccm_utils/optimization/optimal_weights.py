import numpy as np
from scipy.stats import rankdata
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.adam import AdamOptimizer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergence_monitor import ConvergenceMonitor
from tqdm import trange

class WeightOptimizer:
    """
    Gradient-descent optimiser over the weight vector W that combines
    multiple correlation maps into a single convergent map.
    The parent supplies each dataset's arrays; this class scores candidate maps.
    """

    # -----------------------  construction  ---------------------------
    def __init__(self, parent, lr: float = 0.1, max_iters: int = 500):
        self.parent     = parent                     # delegate heavy work
        self.W          = parent.W.copy()            # working copy
        self.best_loss  = -np.inf
        self.best_W     = self.W.copy()

        self.adam               = AdamOptimizer(self.W, lr=lr)
        self.convergence_monitor = ConvergenceMonitor(max_i=max_iters)

        # Spearman scores change only when patient ranks cross. At 10% of a
        # uniform map weight most perturbations moved no patient and the
        # gradient was exactly zero for ~75-85% of maps, so the optimizer
        # stopped where it started. 50% of a uniform weight (0.0094 for 53
        # maps) gives nearly every map a gradient while re-ranking only a few
        # patients per step.
        self.h          = max(1e-5, 0.5 / self.W.size)
        self.iter_weights = []                      # optional compact trace
        self.iter_losses = []
        self.converged  = False
        self._cohort_projections = (
            (parent.precomputed_projections
             if parent.precomputed_projections is not None
             else self._prepare_projections())
            if parent.inner_folds is None else None
        )
        self._inner_projections = (
            self._prepare_inner_projections()
            if parent.inner_folds is not None else None
        )

    # -------------------  helper / utility funcs  ---------------------
    def _tanh_normalize(self, W):        # keep weights bounded ±1 and Σ|w| = 1
        W = np.tanh(W)
        scale = np.sum(np.abs(W))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Cannot normalize zero or nonfinite map weights.")
        return W / scale

    def _clip(self, g, lo=-.5, hi=.5):    # gradient clipping
        return np.clip(g, lo, hi)

    # -----------------------  core maths  -----------------------------
    @staticmethod
    def _broadcast_cosine_similarity(patient_maps, convergent_map):
        """Return patient-by-map cosine similarities for one dataset."""
        maps = np.atleast_2d(convergent_map)
        numerator = patient_maps @ maps.T
        map_norms = np.sqrt(np.einsum('pv,pv->p', maps, maps))
        patient_norms = np.sqrt(
            np.einsum('nv,nv->n', patient_maps, patient_maps)
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            return numerator / (patient_norms[:, None] * map_norms[None, :])

    @staticmethod
    def _spearman_columns(similarities, outcomes):
        """Spearman rho for each similarity column, including tied ranks."""
        ranked_sim = rankdata(similarities, axis=0)
        ranked_y = rankdata(np.asarray(outcomes).reshape(-1))
        ranked_sim -= ranked_sim.mean(axis=0)
        ranked_y -= ranked_y.mean()
        numerator = ranked_y @ ranked_sim
        denominator = np.sqrt(
            np.sum(ranked_y**2) * np.sum(ranked_sim**2, axis=0)
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            return numerator / denominator

    def _rho_array(self, convergent_map):
        """Score each dataset's patient maps against its own outcomes."""
        if self._inner_projections is not None:
            raise ValueError("Out-of-fold scoring requires weights, since each fold has its own map.")
        maps = np.atleast_2d(convergent_map)
        rhos = np.empty((len(self.parent.dataset_names), maps.shape[0]))
        for i, dataset_name in enumerate(self.parent.dataset_names):
            X, y = self.parent.get_dataset(dataset_name)
            similarities = self._broadcast_cosine_similarity(X, maps)
            rhos[i] = self._spearman_columns(similarities, y)
        return rhos[:, 0] if maps.shape[0] == 1 else rhos

    def _prepare_projections(self):
        """Project each scoring cohort onto the fixed component maps once."""
        projections = {}
        n_features = self.parent.MAPS.shape[1]
        rows_per_chunk = max(1, 10_000_000 // n_features)
        for name in self.parent.dataset_names:
            X, y = self.parent.get_dataset(name)
            projected = np.empty((X.shape[0], self.W.size), dtype=float)
            for start in range(0, X.shape[0], rows_per_chunk):
                stop = min(start + rows_per_chunk, X.shape[0])
                chunk = X[start:stop]
                patient_norms = np.sqrt(np.einsum('nv,nv->n', chunk, chunk))
                if np.any(patient_norms == 0):
                    raise ValueError(f"Scoring dataset {name!r} contains a zero-norm X row.")
                projected[start:stop] = (chunk @ self.parent.MAPS.T) / patient_norms[:, None]
            projections[name] = (projected, y)
        return projections

    def _prepare_inner_projections(self):
        """Cache inner-validation products and map Gram matrices per fold.

        Fold maps are only needed during this setup. At every optimizer step,
        the cosine denominator is recomputed for each fold's weighted map;
        omitting it would distort ranks across folds.
        """
        folds = self.parent.inner_folds
        if folds is None:
            raise ValueError("inner_folds must contain at least one fold.")
        prepared = {name: [] for name in self.parent.dataset_names}
        self._inner_reference_stats = {}
        covered = {
            name: np.zeros(self.parent.get_dataset(name)[0].shape[0], dtype=bool)
            for name in self.parent.dataset_names
        }
        n_features = self.parent.MAPS.shape[1]
        for fold_number, fold in enumerate(folds, start=1):
            maps = fold["maps"]
            if set(maps) != set(self.parent.corr_map_names):
                raise ValueError(f"Fold {fold_number} map names differ from the final maps.")
            M = np.stack([
                np.asarray(maps[name], dtype=float).reshape(-1)
                for name in self.parent.corr_map_names
            ])
            if M.shape != self.parent.MAPS.shape or not np.isfinite(M).all():
                raise ValueError(f"Fold {fold_number} maps must be finite and have {n_features} features.")
            norms = np.linalg.norm(M, axis=1)
            if np.any(norms == 0):
                raise ValueError(f"Fold {fold_number} contains a zero-norm map.")
            M /= norms[:, None]
            gram = M @ M.T
            rows_by_dataset = fold["rows"]
            if set(rows_by_dataset) != set(self.parent.dataset_names):
                raise ValueError(f"Fold {fold_number} needs row indices for every scoring dataset.")
            shared_projection = {}
            for name, raw_rows in rows_by_dataset.items():
                X, _ = self.parent.get_dataset(name)
                rows = np.asarray(raw_rows)
                if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
                    raise ValueError(f"Fold {fold_number} rows for {name!r} must be integer indices.")
                if (np.any(rows < 0) or np.any(rows >= len(X))
                        or len(np.unique(rows)) != len(rows)
                        or np.any(covered[name][rows])):
                    raise ValueError(f"Fold {fold_number} repeats or exceeds rows for {name!r}.")
                covered[name][rows] = True
                projection_key = (id(X), rows.tobytes())
                if projection_key in shared_projection:
                    prepared[name].append((rows, shared_projection[projection_key], gram))
                    continue
                projected = np.empty((len(rows), M.shape[0]), dtype=float)
                rows_per_chunk = max(1, 10_000_000 // n_features)
                for start in range(0, len(rows), rows_per_chunk):
                    stop = min(start + rows_per_chunk, len(rows))
                    chunk = X[rows[start:stop]]
                    patient_norms = np.linalg.norm(chunk, axis=1)
                    if np.any(patient_norms == 0):
                        raise ValueError(f"Fold {fold_number} has a zero-norm patient in {name!r}.")
                    projected[start:stop] = (chunk @ M.T) / patient_norms[:, None]
                shared_projection[projection_key] = projected
                if self.parent.inner_score_mode == 'reference_z' and len(rows):
                    # Use training patients' image scores as each fold's
                    # unsupervised reference distribution. This removes map
                    # scale and location offsets without using held-out y.
                    reference_rows = np.ones(len(X), dtype=bool)
                    reference_rows[rows] = False
                    reference = X[reference_rows]
                    if len(reference) < 3:
                        raise ValueError(
                            f"Fold {fold_number} needs three reference patients in {name!r}."
                        )
                    reference_norms = np.linalg.norm(reference, axis=1)
                    if np.any(reference_norms == 0):
                        raise ValueError(
                            f"Fold {fold_number} has a zero-norm reference patient in {name!r}."
                        )
                    reference_projected = (reference @ M.T) / reference_norms[:, None]
                    reference_mean = reference_projected.mean(axis=0)
                    centered = reference_projected - reference_mean
                    reference_cov = centered.T @ centered / len(reference)
                    self._inner_reference_stats[(id(projected), id(gram))] = (
                        reference_mean, reference_cov
                    )
                prepared[name].append((rows, projected, gram))
        for name, row_coverage in covered.items():
            if not row_coverage.all():
                raise ValueError(f"Out-of-fold maps do not score every row of {name!r} exactly once.")
        return prepared

    def predict_inner(self, weights, *, adjusted=False):
        """Return inner-validation scores used to select the weights."""
        if self._inner_projections is None:
            raise ValueError("Inner predictions require inner_folds.")
        candidates = np.atleast_2d(weights)
        predictions = {}
        shared_scores = {}
        for name in self.parent.dataset_names:
            X, _ = self.parent.get_dataset(name)
            scores = np.empty((len(X), candidates.shape[0]), dtype=float)
            for rows, projected, gram in self._inner_projections[name]:
                if not len(rows):
                    continue
                key = (id(projected), id(gram))
                if key not in shared_scores:
                    if adjusted and self.parent.inner_score_mode == 'reference_z':
                        mean, cov = self._inner_reference_stats[key]
                        spread = np.sqrt(np.maximum(
                            np.einsum('bi,ij,bj->b', candidates, cov, candidates), 0
                        ))
                        if not np.isfinite(spread).all() or np.any(spread <= 0):
                            raise ValueError(
                                "A candidate weight vector has no variation among reference patients."
                            )
                        shared_scores[key] = (
                            projected @ candidates.T - (mean @ candidates.T)[None, :]
                        ) / spread[None, :]
                    else:
                        map_norms = np.sqrt(np.einsum('bi,ij,bj->b', candidates, gram, candidates))
                        if not np.isfinite(map_norms).all() or np.any(map_norms <= 0):
                            raise ValueError("A candidate weight vector produces a zero-norm fold map.")
                        shared_scores[key] = (projected @ candidates.T) / map_norms[None, :]
                scores[rows] = shared_scores[key]
            predictions[name] = scores[:, 0] if candidates.shape[0] == 1 else scores
        return predictions

    def _rho_for_weights(self, weights):
        """Score candidate weights using cached patient-by-component products.

        Fixed maps share one norm across patients, which does not affect ranks.
        Inner-fold maps need a separate norm in each fold; predict_inner
        includes that denominator before the complete-cohort Spearman score.
        """
        candidates = np.atleast_2d(weights)
        rhos = np.empty((len(self.parent.dataset_names), candidates.shape[0]))
        if self._inner_projections is not None:
            predictions = self.predict_inner(
                candidates, adjusted=self.parent.inner_score_mode == 'reference_z'
            )
            for i, name in enumerate(self.parent.dataset_names):
                _, y = self.parent.get_dataset(name)
                scores = np.atleast_2d(predictions[name]).T if candidates.shape[0] == 1 else predictions[name]
                rhos[i] = self._spearman_columns(scores, y)
            return rhos[:, 0] if candidates.shape[0] == 1 else rhos
        for i, name in enumerate(self.parent.dataset_names):
            projected, y = self._cohort_projections[name]
            similarities = projected @ candidates.T
            rhos[i] = self._spearman_columns(similarities, y)
        return rhos[:, 0] if candidates.shape[0] == 1 else rhos

    @staticmethod
    def _target(rhos):
        '''Root Mean Squared Error equivalent with Rho'''
        return np.sqrt(np.mean(np.square(rhos), axis=0))

    def _penalty_all(self, thr=0.33, scale=1000):
        return np.sum(1e-6 / (thr - np.abs(self.W))) / scale

    def _penalty_each(self, thr=0.00, k=0.005):
        mask = np.abs(self.W) > thr
        return np.sum(np.abs(self.W) * mask * k)

    # ---------------------  loss + gradient  --------------------------
    def _loss(self, avg_map, penalty=False):
        """Avg map is the current working 'convergent map' to test"""
        rho_arr = self._rho_array(avg_map)
        t       = self._target(rho_arr)
        if penalty:
            t   = t - self._penalty_all() - self._penalty_each()
        return t

    def _loss_for_weights(self, weights):
        loss = self._target(self._rho_for_weights(weights))
        if not np.isfinite(loss).all():
            raise ValueError("Optimization score is undefined for these map weights.")
        return loss

    def _forward_diff_grad(self, base_loss, batch_size=50):
        """
        Evaluate finite-difference perturbations in batches of maps.
        """
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')

        g = np.empty_like(self.W)

        for start in range(0, self.W.size, batch_size):
            stop = min(start + batch_size, self.W.size)
            candidates = np.broadcast_to(self.W, (stop - start, self.W.size)).copy()
            candidates[np.arange(stop - start), np.arange(start, stop)] += self.h

            loss_fwd = np.atleast_1d(self._loss_for_weights(candidates))
            g.flat[start:stop] = (loss_fwd - base_loss) / self.h

        return self._clip(g)

    # -----------------------  main routine  ---------------------------
    def optimise(self, store_iters=False, store_best=False):
        """
        Adam + FD gradient until convergence monitor halts.
        Returns final convergent map.
        """
        self.iter_weights.clear()
        self.iter_losses.clear()
        loss = 0
        bar = trange(self.convergence_monitor.max_iterations, desc="Optimizing Weights")
        for _ in bar:
            if self.converged:
                break
            loss = self._loss_for_weights(self.W)
            if self.parent.inner_folds is not None:
                objective = (
                    "inner-CV calibrated RMS Rho"
                    if self.parent.inner_score_mode == 'reference_z'
                    else "inner-CV raw RMS Rho"
                )
            else:
                objective = "training RMS Rho"
            bar.set_description(f"Optimizing Weights, {objective} = {float(loss):.4f}")
            if store_iters or store_best:
                # The score describes the weights before the Adam step.
                scored_weights = self.W.copy()

            grad = self._forward_diff_grad(loss)
            updated = self._tanh_normalize(self.adam.step(grad))
            self.adam.weights[...] = updated
            self.W = self.adam.weights

            self.converged = self.convergence_monitor.check_convergence(
                weights=self.W, gradient=grad, loss=loss
            )

            if store_best and loss > self.best_loss:
                self.best_loss, self.best_W = loss, scored_weights
            if store_iters:
                self.iter_weights.append(scored_weights)
                self.iter_losses.append(float(loss))

        selected_weights = self.best_W if store_best else self.W
        print("Selected weights:", selected_weights)
        return self.parent._converge_maps(selected_weights)
    
    # ---------- second-stage blend optimiser ---------- #
    def blend_optimize(self, W_opt: np.ndarray, W_unw: np.ndarray, lam_delta: float = 0.05, lam_alpha: float = 0.01, n_grid: int = 200) -> tuple[float, np.ndarray]:
        """Search α∈[0,1] maximising J(alpha) by using switching function to blend optimal weights vs no weighting."""
                   # const
        def J(alpha: float) -> float:
            W = alpha * W_opt + (1 - alpha) * W_unw
            loss = self._loss_for_weights(W)
            penalty = (alpha)**2 * (np.linalg.norm(W_opt - W_unw) ** 2) # Penalizes as alpha increases, which corresponds to larger deviations from W_unw
            return loss - penalty

        alphas = np.linspace(0, 1.0, n_grid)
        j_vals = [J(a) for a in alphas]
        k_best = int(np.argmax(j_vals))
        best_alpha = float(alphas[k_best])
        best_W     = best_alpha * W_opt + (1 - best_alpha) * W_unw
        print(f"Blend optimize: best_alpha={best_alpha}, best_W={best_W}")
        return best_alpha, best_W, self.parent._converge_maps(W=best_W)
