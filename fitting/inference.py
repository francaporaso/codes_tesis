import numpy as np

from fitting.constants import *

class Likelihood:
    def __init__(self, data, model, param_limits, observable='delta_sigma', cov_mode='full'):
        
        self.R = data.R
        self.rhomean = rho_mean(data.redshift)

        if observable=='sigma':
            self.ydata = data.Sigma
            self.cov = data.covS
            
            self.func = model.sigma
        
        elif observable=='delta_sigma':
            self.ydata = data.DSigma_t
            self.cov = data.covDSt

            self.func = model.delta_sigma

        if cov_mode == 'full':
            self.yerr = np.linalg.inv(self.cov)
        elif cov_mode == 'diag':
            # this allows to use log_likelihood with both diag or full covariance!
            self.yerr = np.zeros_like(self.cov)
            np.fill_diagonal(self.yerr, 1/np.diag(self.cov))

        self.limits = param_limits
        self.param_name = list(self.limits.keys())
        

    def log_likelihood(self, theta):
        model = self.func(self.r, *theta)*self.rhomean
        dist = self.y - model
        return -0.5*np.dot(dist, np.dot(self.yerr, dist))

    def log_prior(self, theta):
        ### tener cuidado con el orden de lims!
        if np.prod(
            [self.limits[self.params[j]][0] < theta[j] < self.limits[self.params[j]][1] for j in range(len(self.params))],
            dtype=bool
        ): return 0
        return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

# - easy way to make a joint fit for different data but model with the same parameters.
class JointLikelihood:
    # should be a composition of two or more Likelihood instances
    pass