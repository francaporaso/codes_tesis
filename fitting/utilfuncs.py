import numpy as np

from fitting.models import default_limits

def chi2_red(data, model, invC, ndof):
    d = (data-model)
    return np.sum(np.dot(d, np.dot(invC,d)))/ndof


def make_pos_gaussian(init_guess, NWALKERS, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.array([
        rng.normal(init_guess[i], np.abs(0.15*init_guess[i]), NWALKERS) for i in range(len(init_guess))
    ]).T
    return pos

def validate_pos(pos, model_name):
    rng = np.random.default_rng(seed=0)
    limits = default_limits.get(model_name)
    for i, (lmin, lmax) in enumerate(limits.values()):
        for j, p in enumerate(pos[:,i]):
            if p<lmin or p>lmax:
                print('Invalid pos, redrawing...')
                pos[j,i] = rng.uniform(lmin, lmax)
    return pos