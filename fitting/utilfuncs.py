import numpy as np

def chi2_red(data, model, invC, ndof):
    d = (data-model)
    return np.sum(np.dot(d, np.dot(invC,d)))/ndof

def logistic(x, x0=1, k=10):
    return (1.0+np.exp(-2.0*k*(x-x0)))**(-1)

def make_pos_gaussian(init_guess, NWALKERS, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.array([
        rng.normal(init_guess[i], np.abs(0.15*init_guess[i]), NWALKERS) for i in range(len(init_guess))
    ]).T
    return pos