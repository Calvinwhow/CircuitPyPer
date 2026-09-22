import numpy as np
from tqdm import tqdm
from time import time
from scipy.stats import spearmanr
from calvin_utils.neuroimaging_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.adam import AdamOptimizer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.optimal_weights import WeightOptimizer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergence_monitor import ConvergenceMonitor

class LocalizationOptimizer:
    def __init__(self, corr_map_dict, data_loader, load_in_time=True,
                 mode='unweighted', data_mode=None, map_sample_sizes=None,
                 random_state=None, inner_folds=None,
                 inner_score_mode='raw', precomputed_projections=None):
        """
        corr_map_dict: the dictionary containing underlying Rho maps.
        data_loader: object which handles loading of the underlying npy arrays.
        data_mode: 'memmap' opens each dataset once as a read-only memory map;
            'ram' loads each dataset once into memory. Defaults to 'memmap'.
        load_in_time: legacy option; True selects 'memmap' and False selects
            'ram' when data_mode is not given.
        mode: initial map weights: unweighted | gaussian | weighted.
        map_sample_sizes: sizes for the component maps when mode='weighted';
            independent of the scoring datasets in data_loader.
        random_state: optional seed for reproducible Gaussian initialization.
        inner_folds: regression maps and validation row indices for inner
            weight selection. corr_map_dict contains the maps rebuilt on the
            complete outer-training set for the resulting fitted map.
        inner_score_mode: 'raw' pools inner-fold cosines directly;
            'reference_z' calibrates each inner-validation score against the
            corresponding inner-training patient scores.
        precomputed_projections: optional fixed-map patient-by-component
            products supplied by the I/O layer to reuse them across CV folds.
        """

        self.data_loader = data_loader
        self.dataset_names = tuple(data_loader.dataset_names_list)
        if not self.dataset_names:
            raise ValueError("At least one scoring dataset is required.")
        self.map_sample_sizes = map_sample_sizes
        self.random_state = random_state
        self.inner_folds = inner_folds
        self.precomputed_projections = precomputed_projections
        if inner_score_mode not in {'raw', 'reference_z'}:
            raise ValueError("inner_score_mode must be 'raw' or 'reference_z'.")
        self.inner_score_mode = inner_score_mode
        self._rng = np.random.default_rng(random_state) if random_state is not None else None
        self.data_mode = (
            'memmap' if load_in_time else 'ram'
        ) if data_mode is None else data_mode
        if self.data_mode not in {'memmap', 'ram'}:
            raise ValueError("data_mode must be 'memmap' or 'ram'.")
        self.best_loss = 0
        self.converged = False
        self.corr_map_names = [k for k in corr_map_dict.keys()]
        self.corr_map_dict = self._clean_corr_dict(corr_map_dict)
        self.W = self._initialize_weights(mode)
        self.MAPS = self._initialize_maps(self.corr_map_dict)
        self.datasets = self._init_data()
        if self.precomputed_projections is not None:
            if self.inner_folds is not None or set(self.precomputed_projections) != set(self.dataset_names):
                raise ValueError("Precomputed projections need every fixed-map scoring dataset.")
            for name, (projected, outcomes) in self.precomputed_projections.items():
                X, y = self.get_dataset(name)
                if (np.asarray(projected).shape != (len(X), self.MAPS.shape[0])
                        or np.asarray(outcomes).reshape(-1).shape != (len(y),)):
                    raise ValueError(f"Precomputed projections for {name!r} have wrong shape.")
        self._readout()
        self.engine = WeightOptimizer(self)
        self.adam = AdamOptimizer(self.W, lr=0.1)
        self.convergence_monitor = ConvergenceMonitor(max_i=500)
        
    ### Preprocessing ###
    def _clean_corr_dict(self, corr_map_dict):
        cleaned = {}
        for name, matrix in corr_map_dict.items():
            values = np.asarray(matrix, dtype=float).reshape(-1)
            if values.size == 0:
                raise ValueError(f"Component map {name!r} is empty.")
            cleaned[name] = CorrelationCalculator._check_for_nans(
                values, nanpolicy='remove', verbose=False
            )
        return cleaned
    
    def _handle_nans(self, arr):
        return CorrelationCalculator._check_for_nans(arr, nanpolicy='remove', verbose=False)

    ### I/O ###
    def _initialize_maps(self, corr_map_dict):
        """Return component maps as (maps, features), in weight order."""
        if not self.corr_map_names:
            raise ValueError("At least one component map is required.")
        n_vox = corr_map_dict[self.corr_map_names[0]].size
        M = np.empty((len(self.corr_map_names), n_vox), dtype=float)
        for i, name in enumerate(self.corr_map_names):
            values = corr_map_dict[name]
            if values.size != n_vox:
                raise ValueError(
                    f"Component map {name!r} has {values.size} features; expected {n_vox}."
                )
            # Unit L2 per map. Raw maps differ in scale (LNM t-maps ~3x), so a
            # fixed weight step moved big maps and not small ones; normalized, a
            # step is the same share of every map and a weight reads directly as
            # that map's share of the convergent map.
            norm = np.linalg.norm(values)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError(f"Component map {name!r} has zero or nonfinite norm.")
            M[i] = values / norm
        return M

    def _initialize_weights(self, mode):
        '''
        Initializes weights as a numpy array.
        Shape: (1, N) where N = number of correlation maps.
        mode: 'weighted', 'unweighted', or 'gaussian'.
        '''
        n_maps = len(self.corr_map_names)
        weights = np.zeros(n_maps)  # shape: (N,)
        for i, k in enumerate(self.corr_map_names):
            if mode == 'weighted':
                if self.map_sample_sizes is not None and k in self.map_sample_sizes:
                    weights[i] = float(self.map_sample_sizes[k])
                elif k in self.dataset_names:
                    weights[i] = self.data_loader.load_dataset(k)['niftis'].shape[0]
                else:
                    raise ValueError(
                        "Weighted initialization derives each map's sample size "
                        f"from its same-named dataset; no dataset matches {k!r}."
                    )
                if not np.isfinite(weights[i]) or weights[i] <= 0:
                    raise ValueError(f"Map sample size for {k!r} must be positive.")
            elif mode == 'gaussian':
                weights[i] = (
                    np.random.normal(0.1, 1) if self.random_state is None
                    else self._rng.normal(0.1, 1)
                )
            elif mode == 'unweighted':
                weights[i] = 1.0
            else:
                raise ValueError(f"Unknown weight initialization mode: {mode!r}.")
        weights = np.reshape(weights, (1, n_maps))   # reshape weights to shape: (1, N)
        scale = np.sum(np.abs(weights))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("Initial map weights must have a finite nonzero total magnitude.")
        return weights / scale
    
    @staticmethod
    def _require_finite(array, dataset_name, key):
        """Scan mapped arrays in bounded chunks without materializing them."""
        if array.ndim == 0:
            raise ValueError(f"{dataset_name!r} {key} must have an observation axis.")
        row_width = int(np.prod(array.shape[1:]))
        rows_per_chunk = max(1, 1_000_000 // max(1, row_width))
        for start in range(0, array.shape[0], rows_per_chunk):
            if not np.isfinite(array[start:start + rows_per_chunk]).all():
                raise ValueError(
                    f"{dataset_name!r} {key} contains NaN or infinity. "
                    "Memory-mapped data must be cleaned before optimization."
                )

    def _init_data(self):
        """Open each scoring dataset once in the selected storage mode."""
        datasets = {}
        checked_arrays = {}
        mmap_mode = 'r' if self.data_mode == 'memmap' else None
        for k in self.dataset_names:
            data = self.data_loader.load_dataset(k, mmap_mode=mmap_mode)
            for key in ('niftis', 'indep_var'):
                original = data[key]
                if id(original) in checked_arrays:
                    data[key] = checked_arrays[id(original)]
                    continue
                if self.data_mode == 'memmap':
                    self._require_finite(original, k, key)
                    checked_arrays[id(original)] = original
                else:
                    cleaned = CorrelationCalculator._check_for_nans(
                        original, nanpolicy='remove', verbose=False
                    )
                    checked_arrays[id(original)] = cleaned
                    data[key] = cleaned
            X, y = data['niftis'], data['indep_var']
            if X.ndim != 2 or X.shape[1] != self.MAPS.shape[1]:
                raise ValueError(
                    f"Scoring dataset {k!r} has X shape {X.shape}; expected "
                    f"(patients, {self.MAPS.shape[1]})."
                )
            if y.shape not in {(X.shape[0],), (X.shape[0], 1)}:
                raise ValueError(
                    f"Scoring dataset {k!r} has y shape {y.shape}; expected "
                    f"({X.shape[0]},) or ({X.shape[0]}, 1)."
                )
            if X.shape[0] < 3 or np.unique(y).size < 2:
                raise ValueError(
                    f"Scoring dataset {k!r} needs at least three patients and varying y."
                )
            datasets[k] = data
        return datasets

    def get_dataset(self, dataset_name):
        """Return the cached patient matrix and outcome vector."""
        data = self.datasets[dataset_name]
        return data['niftis'], data['indep_var']
    
    def _readout(self):
        print(f"===Imaging Optimizer Initialized ({self.data_mode})===")
        print(f"Initializing: \n\t Weights: {self.W.shape}  \n\t Training Maps: {self.MAPS.shape}")
        print("===        Component Maps        ===")
        for k in self.corr_map_names[:10]:
            print(f"\t Map: {k}")
        if len(self.corr_map_names) > 10:
            print(f"\t ... and {len(self.corr_map_names) - 10} more maps")
        print("===        Scoring Data        ===")
        for k in self.dataset_names:
            print(f"\t Dataset: {k}")
    
    ### Nifti Functions ### 
    def _converge_maps(self, W=None):
        '''
        Calculate the convergent map. 
        Allow weights to be passed as an argument (for perturbation).
        If no weight argument, will use the weights object (for optimization)
        '''
        if W is None:           
            W = self.W
        return W @ self.MAPS
    
    def _broadcast_cosine_similarity(self, patient_maps, convergent_map):
        """Cosine similarity for one or more convergent maps.

        Return one score per patient for a single map, or a patient-by-map
        matrix for a batch of maps.
        """
        maps = np.atleast_2d(convergent_map)
        numerator = patient_maps @ maps.T
        map_norms = np.sqrt(np.einsum('ij,ij->i', maps, maps))
        patient_norms = np.sqrt(
            np.einsum('ij,ij->i', patient_maps, patient_maps)
        )
        similarities = numerator / (patient_norms[:, None] * map_norms[None, :])
        return similarities[:, 0] if maps.shape[0] == 1 else similarities
    
    def _calculate_similarity(self, patient_maps, convergent_map):
        """Orchestrate similarity calculation"""
        return self._broadcast_cosine_similarity(patient_maps, convergent_map)
    
    ### Public ###
    def optimize(self, second_stage=False, **kwargs):
        self.blended_map = None
        self.optimized_map = self.engine.optimise(store_best=True, **kwargs)
        if second_stage:
            self.alpha, self.W_final, self.blended_map = self.engine.blend_optimize(
                        W_opt=self.engine.best_W,
                        W_unw=self._initialize_weights('unweighted'),
                        lam_delta=0.05,
                        lam_alpha=0.01)
        return self.optimized_map, self.blended_map
    

# Preserve existing notebook and script imports.
NiftiOptimizer = LocalizationOptimizer
