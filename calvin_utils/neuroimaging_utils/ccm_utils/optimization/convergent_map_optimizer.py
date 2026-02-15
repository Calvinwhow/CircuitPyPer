import numpy as np
from tqdm import tqdm
from time import time
from scipy.stats import spearmanr
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.ccm_utils.optimization.adam import AdamOptimizer
from calvin_utils.ccm_utils.optimization.optimal_weights import WeightOptimizer
from calvin_utils.ccm_utils.optimization.convergence_monitor import ConvergenceMonitor

class NiftiOptimizer:
    def __init__(self, corr_map_dict, data_loader, load_in_time=True):
        self.data_loader = data_loader
        self.best_loss = 0
        self.iter_maps = []
        self.converged = False
        self.corr_map_names = [k for k in corr_map_dict.keys()]
        self.corr_map_dict = self._clean_corr_dict(corr_map_dict)
        self.W = self._initialize_weights('gaussian')
        self.MAPS = self._initialize_maps(self.corr_map_dict)
        self.datasets = self._init_data(load_in_time)
        self._readout()
        self.engine = WeightOptimizer(self)
        self.adam = AdamOptimizer(self.W, lr=0.1)
        self.convergence_monitor = ConvergenceMonitor(max_i=500)
        
    ### Preprocessing ###
    def _clean_corr_dict(self, corr_map_dict):
        for dataset, matrix in corr_map_dict.items():
            corr_map_dict[dataset] = CorrelationCalculator._check_for_nans(matrix, nanpolicy='remove', verbose=False)
        return corr_map_dict
    
    def _handle_nans(self, arr):
        return CorrelationCalculator._check_for_nans(arr, nanpolicy='remove', verbose=False)

    ### I/O ###
    def _initialize_maps(self, corr_map_dict):
        '''Returns the niftis as a flattened array. Shape (N, Voxels)'''
        n_obs = self.W.shape[1]
        n_vox = len(corr_map_dict[self.corr_map_names[0]].flatten())
        M = np.zeros((n_obs, n_vox))
        for i, k in enumerate(corr_map_dict.keys()):
            M[i,:] = corr_map_dict[k]
        return M

    def _initialize_weights(self, mode='unweighted'):
        '''
        Initializes weights as a numpy array.
        Shape: (1, N) where N = number of correlation maps.
        mode: 'weighted', 'unweighted', or 'gaussian'
        '''
        n_maps = len(self.corr_map_names)
        weights = np.zeros(n_maps)  # shape: (N,)
        for i, k in enumerate(self.corr_map_names):
            if mode == 'weighted':
                data = self.data_loader.load_dataset(k)
                weights[i] = data['niftis'].shape[0]
            elif mode == 'gaussian':
                weights[i] = np.random.normal(0.1, 1)
            else:  # 'unweighted'
                weights[i] = 1.0
        weights = np.reshape(weights, (1, n_maps))   # reshape weights to shape: (1, N)
        return weights / np.sum(weights)
    
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
        print("===Imaging Optimizer Initialized===")
        print(f"Initializing: \n\t Weights: {self.W.shape}  \n\t Training Maps: {self.MAPS.shape}")
        print("===        Training Data        ===")
        for k in self.corr_map_dict.keys():
            print(f"\t Training Dataset (For weighted average): {k}")
        print("===         Testing Data        ===")
        for k in self.data_loader.dataset_names_list:
            print(f"\t Testing Dataset (To predict outcomes using weighted average): {k}")
    
    ### Nifti Functions ### 
    def _converge_maps(self, W=None, verbose=False):
        '''
        Calculate the convergent map. 
        Allow weights to be passed as an argument (for perturbation).
        If no weight argument, will use the weights object (for optimization)
        '''
        if W is None:           
            W = self.W
        return W @ self.MAPS
    
    def _broadcast_cosine_similarity(self, patient_maps, convergent_map):
        '''broadcast matmul of patient maps against convergent map'''
        NUM = np.matmul(patient_maps,convergent_map.T)    #shape (n_obs,  ) <-  (n_obs, n_vox) @ (n_vox, 1)
        A = np.sqrt(np.sum(convergent_map**2, axis=1))    #shape (n_obs,  )   <-  (1, n_vox) summed over n_vox
        B = np.sqrt(np.sum(patient_maps**2,   axis=1))     #shape (n_obs,  )   <-  (n_obs, n_vox) summed over n_vox
        DEN = A*B
        CS =  NUM.flatten() / DEN
        return CS
    
    def _calculate_similarity(self, patient_maps, convergent_map):
        """Orchestrate similarity calculation"""
        return self._broadcast_cosine_similarity(patient_maps, convergent_map)
    
    ### Public ###
    def optimize(self, **kwargs):
        self.optimized_map = self.engine.optimise(**kwargs)
        self.alpha, self.W_final, self.blended_map = self.engine.blend_optimize(
                        W_opt=self.engine.best_W,
                        W_unw=self._initialize_weights('unweighted'),
                        lam_delta=0.05,
                        lam_alpha=0.01)
        return self.optimized_map, self.blended_map
    
       