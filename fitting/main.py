from multiprocessing import Pool
import emcee

from fitting.constants import *
from fitting.inference import *
from fitting.io import *
from fitting.models import *
from fitting.utilfuncs import *


def run_emcee(NCORES, NIT, NWALKERS, data_filename, save_filename, model_name, observable, cov_mode):
    data = read_dataprofile_fits(name=data_filename)

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name),
        param_limits=default_limits.get(model_name),
        observable=observable,
        cov_mode=cov_mode
    )

    init_pos = make_pos_gaussian(
        init_guess=default_guess.get(model_name),
        NWALKERS=NWALKERS,
        seed=0
    )
    validate_pos(init_pos, model_name)
    
    group_name = f'emcee/{model_name}/{cov_mode}'
    backend = emcee.backends.HDFBackend(save_filename, name=group_name)
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS, L.nparams, L.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(init_pos, NIT, progress=True, store=True)

    return sampler

if __name__ == '__main__':
    from fitting.plotting import *
    
    sampler = run_emcee(
        6,1000,32,
        data_filename='lensing/results/lensing_EUC-PURE_MICE_N30_Rv06-10_z020-040_typeS_binlin.fits',
        save_filename='fitting/results/fitting_EUC-PURE_MICE_N30_Rv06-10_z020-040_typeS_binlin.hdf5',
        model_name='B15',
        observable='delta_sigma',
        cov_mode='full'
    )
    plot_chains(sampler.get_chain())
    plt.show()

    plot_corner(sampler);
    plt.show()
