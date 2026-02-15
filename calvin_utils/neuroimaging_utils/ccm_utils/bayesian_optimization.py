import numpy as np
from scipy.stats import spearmanr
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from skopt import gp_minimize

class NiftiBayesianOptimizer:
    def __init__(self, corr_map_dict, data_loader, load_in_time=True):
        self.data_loader = data_loader
        self.corr_map_names = list(corr_map_dict.keys())
        self.corr_map_dict = self._handle_nans(corr_map_dict)
        self.MAPS = self._initialize_maps(self.corr_map_dict)
        self.datasets = self._init_data(load_in_time)
        self._readout()

    def _handle_nans(self, corr_map_dict):
        for dataset, matrix in corr_map_dict.items():
            corr_map_dict[dataset] = CorrelationCalculator._check_for_nans(matrix, nanpolicy='remove', verbose=False)
        return corr_map_dict

    def _initialize_maps(self, corr_map_dict):
        n_obs = len(self.corr_map_names)
        n_vox = len(corr_map_dict[self.corr_map_names[0]].flatten())
        M = np.zeros((n_obs, n_vox))
        for i, k in enumerate(self.corr_map_names):
            M[i, :] = corr_map_dict[k]
        return M

    def _init_data(self, load_in_time=False):
        if load_in_time:
            return None
        datasets = {}
        for k in self.data_loader.dataset_names_list:
            datasets[k] = self.data_loader.load_dataset(k)
            datasets[k]['niftis'] = CorrelationCalculator._check_for_nans(datasets[k]['niftis'], nanpolicy='remove', verbose=False)
            datasets[k]['indep_var'] = CorrelationCalculator._check_for_nans(datasets[k]['indep_var'], nanpolicy='remove', verbose=False)
        return datasets

    def _readout(self):
        print("=== Bayesian Imaging Optimizer Initialized ===")
        print(f"Training Maps Shape: {self.MAPS.shape}")
        print("Training Datasets:")
        for k in self.corr_map_names:
            print(f"\t{k}")
        print("Testing Datasets:")
        for k in self.data_loader.dataset_names_list:
            print(f"\t{k}")

    def _converge_maps(self, W):
        convergent_map = W @ self.MAPS
        return CorrelationCalculator._check_for_nans(convergent_map, nanpolicy='remove', verbose=False)

    def _broadcast_cosine_similarity(self, patient_maps, convergent_map):
        NUM = patient_maps @ convergent_map.T
        DEN = np.linalg.norm(patient_maps, axis=1) * np.linalg.norm(convergent_map)
        return (NUM.flatten() / DEN)

    def _calculate_similarity(self, patient_maps, convergent_map):
        return self._broadcast_cosine_similarity(patient_maps, convergent_map)

    def _rho_function(self, AVG_MAP):
        RHO_LIST = []
        for k in self.data_loader.dataset_names_list:
            if self.datasets is None:
                dataset = self.data_loader.load_dataset(k)
                niftis = CorrelationCalculator._check_for_nans(dataset['niftis'], nanpolicy='remove', verbose=False)
                Y = CorrelationCalculator._check_for_nans(dataset['indep_var'], nanpolicy='remove', verbose=False)
            else:
                niftis = self.datasets[k]['niftis']
                Y = self.datasets[k]['indep_var']

            X = self._calculate_similarity(niftis, AVG_MAP)
            RHO, _ = spearmanr(X, Y)
            RHO_LIST.append(RHO)
        return np.array(RHO_LIST)

    def _target_function(self, RHO_ARR):
        return np.sqrt(np.mean(np.abs(RHO_ARR)))

    def _penalty_all_weights(self, W, threshold=0.33, downscale=1000):
        p = np.sum(1e-6 / (threshold - np.abs(W)))
        return p / downscale

    def _loss_function(self, W):
        AVG_MAP = self._converge_maps(W)
        RHO_ARR = self._rho_function(AVG_MAP)
        T = self._target_function(RHO_ARR)
        P = self._penalty_all_weights(W)
        return -(T - P)

    def optimize(self, n_calls=50):
        bounds = [(0.0, 1.0)] * len(self.corr_map_names)

        def objective(W):
            W = np.array(W)
            W /= np.sum(W)
            return self._loss_function(W)

        print("Starting Bayesian Optimization...")
        result = gp_minimize(
            objective,
            dimensions=bounds,
            n_calls=n_calls,
            acq_func='EI',
            n_initial_points=10,
            random_state=42,
            verbose=True
        )

        optimal_W = np.array(result.x)
        optimal_W /= np.sum(optimal_W)
        print(f"Optimization complete. Best Weights: {optimal_W}")

        return self._converge_maps(optimal_W)
