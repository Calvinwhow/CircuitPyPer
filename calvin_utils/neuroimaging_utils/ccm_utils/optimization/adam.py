import numpy as np

class AdamOptimizer:
    def __init__(self, weights, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        '''
        weights - for optimization
        lr - learning rate
        beta1 - moment bias scaling factor
        beta2 - variance bias scaling factor
        epsilon - negligible small number to avoid div by zero
        m - first moment vector (mean of gradient)
        v - second moment vector (variance of gradient)
        t - time step
        '''
        self.weights = weights
        self.lr = lr
        self.beta1 = beta1
        self.beta2  = beta2
        self.epsilon = epsilon
        self.m = np.zeros(weights.shape)
        self.v = np.zeros(weights.shape)
        self.t = 0
    
    ### Internal Orchestrator ###
    def _update(self, gradient):
        '''Performs a maximizing step by ADDING the gradient (direction of steepest ascent w.r.t. loss) to the prior weights.'''
        self.t += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * gradient            # momentum
        self.v = self.beta2 * self.v + (1-self.beta2) * (gradient**2)       # variance
        m_hat = self.m / (1 - self.beta1 ** self.t)                         # bias corrected momentum (for time step)
        v_hat = self.v / (1 - self.beta2 ** self.t)                         # bias corrected variance (for time step)
        self.weights += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)   # update the weights 
    
    ### Public API ###
    def step(self, gradient):
        self._update(gradient)
        return self.weights