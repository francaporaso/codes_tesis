from multiprocessing import Pool
import emcee
import h5py

from fitting.constants import *
from fitting.inference import *
from fitting.io import *
from fitting.models import *
from fitting.utilfuncs import *


def run_emcee(
        NCORES, NIT, NWALKERS, 
        data_filename, save_filename, model_name, observable, cov_mode,
        init_guess):
    
    data = read_dataprofile_fits(name=data_filename)

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name)(data.redshift),
        param_limits=default_limits.get(model_name),
        observable=observable,
        cov_mode=cov_mode
    )

    init_pos = make_pos_gaussian(
        init_guess=init_guess,
        NWALKERS=NWALKERS,
        seed=0
    )
    validate_pos(init_pos, model_name)
    
    group_name = f'emcee/{model_name}/{observable}/{cov_mode}'
    backend = emcee.backends.HDFBackend(save_filename, name=group_name)
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS, L.nparams, L.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(init_pos, NIT, progress=True, store=True)

    return sampler

if __name__ == '__main__':
    from fitting.plotting import *
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--observable', type=str, default='delta_sigma', action='store', choices=['delta_sigma', 'sigma'])
    parser.add_argument('--model', type=str, default='HSW', action='store', choices=['HSW', 'mLW', 'TH'])
    args = parser.parse_args()

    NCORES = 32
    NIT = 5000
    NWALKERS = 64
    PLOT = False #True

    sample = 'full'

    model_name = args.model
    observable = args.observable
    cov_mode = 'full'

    if observable=='delta_sigma':
        init_guess = default_guess.get(model_name)[:-1]
    else:
        init_guess = default_guess.get(model_name)

    for i, rv in enumerate(['06-10', '10-50']):
        for j, t in enumerate(['mixed', 'S', 'R']):

            data_filename = f'lensing/results/lensing_rev2_MICE_N30_Rv{rv}_z020-040_type{t}_binlin.fits'
            chain_filename = f'fitting/results/fitting_MICE_rev2-{sample}_N30_Rv{rv}_z020-040_type{t}_binlin.hdf5'

            sampler = run_emcee(
                NCORES=NCORES,NIT=NIT,NWALKERS=NWALKERS,
                data_filename=data_filename,
                save_filename=chain_filename,
                model_name=model_name,
                observable=observable,
                cov_mode=cov_mode,
                init_guess=init_guess
            )

            param_names = list(default_limits.get(model_name).keys())
            # not possible to fix params for now
            fitpar, errpar = get_fitted_params(
                sampler.get_chain(discard=int(NIT*0.40)), 
                param_names
            )

            with h5py.File(chain_filename, 'a') as f:
                group_path = f'fitedparams/{model_name}/{observable}/{cov_mode}'

                # Overwrite if exists
                if group_path in f:
                    del f[group_path]

                grp = f.create_group(group_path)

                for pname in param_names:
                    pgrp = grp.create_group(pname)
                    pgrp.create_dataset('median', data=fitpar[pname])
                    pgrp.create_dataset('errs', data=np.array(errpar[pname]))


            if PLOT:
                plot_chains(sampler.get_chain())
                plt.show()

                plot_corner(sampler, discard=int(NIT*0.40));
                plt.show()
