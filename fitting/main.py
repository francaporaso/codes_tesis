from multiprocessing import Pool
from argparse import ArgumentParser
import emcee
import h5py
import numpy as np
import matplotlib.pyplot as plt
import toml

#from fitting.constants import rho_mean
from fitting.inference import Likelihood
from fitting.io import read_dataprofile_fits
from fitting.models import models_dict, default_limits, default_guess
from fitting.utilfuncs import check_output_exists, get_fitted_params, validate_pos, make_pos, chi2_red
from fitting.plotting import plot_chains, plot_corner

def run_emcee(
        NCORES, NIT, NWALKERS, 
        data_filename, save_filename, model_name, observable, cov_mode,
        init_guess, pos_dist, seed):
    
    data = read_dataprofile_fits(name=data_filename)

    L = Likelihood(
        data=data,
        model=models_dict.get(model_name)(data.redshift),
        param_limits=default_limits.get(model_name),
        observable=observable,
        cov_mode=cov_mode
    )

    init_pos = make_pos(
        init_guess=init_guess,
        NWALKERS=NWALKERS,
        seed=seed,
        dist=pos_dist,
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


class Config:
    def __init__(self, configfile):
        cfg = toml.load(configfile)

        self.data : dict = {
            'folder' : cfg['data']['folder'],
            'prefix' : cfg['data']['prefix']
        }
        self.chain : dict = {
            'folder' : cfg['chain']['folder'],
            'prefix' : self.get_prefix()
        }

        self.rv_ranges : list[str] = cfg['data']['rv_ranges']
        self.z_ranges : list[str] = cfg['data']['z_ranges']
        self.voidtypes : list[str] = cfg['data']['voidtypes']
        self.binning : str = cfg['data']['binning']

        self.ncores : int = cfg['run']['ncores']
        self.nit : int = cfg['run']['nit']
        self.nwalkers : int = cfg['run']['nwalkers']
        self.do_plot : bool = cfg['run']['do_plot']
        self.overwrite : bool = cfg['run']['overwrite']

        self.cov_mode : str = cfg['fit']['cov_mode']
        self.observables : list = cfg['fit']['observables']
        self.models : list = cfg['fit']['models']
        self.pos_dist : str = cfg['fit']['pos_dist']
        self.seed : int = cfg['fit']['seed']
        self.discardp : float = cfg['fit']['discardp']

    def get_prefix(self):
        return '_'.join(self.data['prefix'].split('_')[1:])
        

def main():

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.toml', action='store')
    args = parser.parse_args()

    cfg = Config(args.config)

    Total = len(cfg.models)*len(cfg.observables)*len(cfg.z_ranges)*len(cfg.rv_ranges)*len(cfg.voidtypes)
    print(f' >> Fitting {len(cfg.models)} model(s) x {len(cfg.observables)} profile(s) x {len(cfg.z_ranges)} redshift bin(s) x {len(cfg.rv_ranges)} radius bin(s) x {len(cfg.voidtypes)} void type(s)')
    print(f' >> {Total=} \n')

    i = 0
    for model in cfg.models:
        for obs in cfg.observables:

            if obs=='delta_sigma':
                init_guess = default_guess.get(model)[:-1]
            elif obs=='sigma':
                init_guess = default_guess.get(model)

            for redshift in cfg.z_ranges:
                for rv in cfg.rv_ranges:
                    for vt in cfg.voidtypes:

                        i+=1
                        print(f'\n [{i}/{Total}]')

                        data_filename = f'{cfg.data["folder"]}{cfg.data["prefix"]}_Rv{rv}_z{redshift}_type{vt}_bin{cfg.binning}.fits'
                        chain_filename = f'{cfg.chain["folder"]}fitting_{cfg.chain["prefix"]}_Rv{rv}_z{redshift}_type{vt}_bin{cfg.binning}.hdf5'

                        assert check_output_exists(chain_filename, overwrite=cfg.overwrite)

                        #print(f' >> Fitting {data_filename}')
                        #print(f' >> Saving to {chain_filename}')

                        print('-'*15)
                        print(f' Model: {model}')
                        print(f' Profile: {obs}')
                        print(f' z: {redshift}')
                        print(f' Rv: {rv}')
                        print(f' Type: {vt}')
                        print('-'*15)

                        sampler = run_emcee(
                            NCORES=cfg.ncores,NIT=cfg.nit,NWALKERS=cfg.nwalkers,
                            data_filename=data_filename,
                            save_filename=chain_filename,
                            model_name=model,
                            observable=obs,
                            cov_mode=cfg.cov_mode,
                            init_guess=init_guess,
                            pos_dist=cfg.pos_dist,
                            seed=cfg.seed,
                        )

                        param_names = list(default_limits.get(model).keys())
                        # not possible to fix params for now
                        discard = int(cfg.nit * cfg.discardp)
                        fitpar, errpar = get_fitted_params(
                            sampler.get_chain(discard=discard), 
                            param_names
                        )
                        
                        # print result values from fit
                        #print(f'>> model: {model} | prof: {obs} | rv: {rv} | z:{redshift} | type: {vt}')
                        # TODO: incorporate the chi2 to the file...
                        # print(f'chi^2 = {chi2_red()}')
                        print(' Fitted params:')
                        for (key, value), e in zip(fitpar.items(), errpar.values()):
                            print(f'    > {key} = {value:.4g} ± ({e[0]:.4g},{e[1]:.4g})')

                        with h5py.File(chain_filename, 'a') as f:
                            group_path = f'fitedparams/{model}/{obs}/{cfg.cov_mode}'

                            # Overwrite if exists
                            if group_path in f:
                                del f[group_path]

                            grp = f.create_group(group_path)

                            for pname in param_names:
                                pgrp = grp.create_group(pname)
                                pgrp.create_dataset('median', data=fitpar[pname])
                                pgrp.create_dataset('errs', data=np.array(errpar[pname]))


                        if cfg.do_plot:
                            plot_chains(sampler.get_chain(), labels=list(fitpar.keys()))
                            plt.show()

                            plot_corner(sampler, discard=discard);
                            plt.show()

if __name__ == '__main__':
    from time import time
    print(' Start '.center(15, '#'))
    tini = time()
    main()
    print(' End :) '.center(15, '#'))
    print(f' >> Took {(time()-tini)/60.0} min <<\n')
