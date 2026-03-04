from multiprocessing import Pool
import emcee

from fitting.constants import *
from fitting.inference import *
from fitting.io import *
from fitting.models import *
from fitting.utilfuncs import *

def run_emcee(NCORES, pos, NIT, L):
    nwalkers, nparams = pos.shape

    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, nparams, L.log_probability, pool=pool
        )
        sampler.run_mcmc(pos, NIT, progress=True)

    return sampler

if __name__ == '__main__':
    from fitting.plotting import *
    
    sampler = run_emcee()
    plot_chains(sampler);
    plot_corner(sampler);

    