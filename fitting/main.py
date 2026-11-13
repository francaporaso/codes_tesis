from multiprocessing import Pool
from argparse import ArgumentParser
import emcee
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from fitting.settings import Settings, FileClassifier
from fitting.constants import get_cosmo
from fitting.inference import Likelihood
from fitting.io import read_dataprofile_fits
from fitting.models import models_dict, default_limits, default_guess
from fitting.utilfuncs import (
    # check_output_exists,
    check_group_exists,
    get_fitted_params,
    validate_pos,
    make_pos,
    chi2_red,
)
from fitting.plotting import plot_chains, plot_corner, plot_profile

import os

os.environ["OMP_NUM_THREADS"] = "1"


def nested_dict():
    return defaultdict(nested_dict)


def run_emcee(
    cosmo,
    NCORES: int,
    NIT: int,
    NWALKERS: int,
    burn_in: int,
    moves: list[dict],
    data_filename: str,
    save_filename: str,
    overwrite: bool,
    model_name: str,
    observable: str,
    cov_mode: str,
    limits: dict,
    fixed: dict,
    init_guess: list[float],
    pos_dist: str,
    seed: int,
):

    group_name = f"emcee/{model_name}/{observable}/{cov_mode}"
    assert check_group_exists(save_filename, group_name, overwrite)

    data = read_dataprofile_fits(filename=data_filename)

    L = Likelihood(
        data=data,
        model=models_dict[model_name](cosmo, data.redshift),
        param_limits=limits,
        fixed_params=fixed,
        observable=observable,
        cov_mode=cov_mode,
    )

    init_pos = make_pos(
        init_guess=init_guess,
        NWALKERS=NWALKERS,
        seed=seed,
        dist=pos_dist,
    )
    assert init_pos.shape[1] == L.nparams, f"{init_pos.shape}!={L.nparams}"
    # init_pos = validate_pos(init_pos, L.param_name, L.limits, seed=seed)
    moves_list = [(emcee.moves.__dict__[m["name"]](), m["weight"]) for m in moves]

    backend = emcee.backends.HDFBackend(str(save_filename), name=group_name)
    with Pool(processes=NCORES) as pool:
        sampler = emcee.EnsembleSampler(
            NWALKERS,
            L.nparams,
            L.log_probability,
            pool=pool,
            backend=backend,
            moves=moves_list,
        )
        print(" >> running burn in")
        state = sampler.run_mcmc(init_pos, burn_in, progress=True, store=False)
        sampler.reset()
        print(" >> running emcee")
        sampler.run_mcmc(state, NIT, progress=True, store=True)

    return sampler, L


def main():

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.toml", action="store")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = Settings(args.config)

    # TODO add the fixed in the config file
    fixed = {"sigma0": 0.0}

    # get files with label inside folder
    files_list = []
    for f in Path(cfg.data["folder"]).expanduser().rglob(f"*{cfg.data['label']}*"):
        files_list.append(FileClassifier(*f.name.split("_")))
    RVS = sorted(list(set(fi.rv_range for fi in files_list)), key=lambda s: s[2:])
    ZS = sorted(list(set(fi.z_range for fi in files_list)), key=lambda s: s[1:])
    TYPES = sorted(list(set(fi.voidtype for fi in files_list)), reverse=True)

    # names saved in dict to easier retrival
    p = nested_dict()
    for f in files_list:
        p[f.rv_range][f.z_range][f.voidtype] = Path(
            cfg.data["folder"] + f.filename
        ).expanduser()

    Total = len(cfg.models) * len(cfg.observables) * len(ZS) * len(RVS) * len(TYPES)
    print(
        (
            f" >> Fitting {len(cfg.models)} model(s) "
            f"x {len(cfg.observables)} profile(s) "
            f"x {len(ZS)} redshift bin(s) "
            f"x {len(RVS)} radius bin(s) "
            f"x {len(TYPES)} void type(s)"
        )
    )
    print(f" >> {Total=}")

    i = 0
    for model in cfg.models:
        for obs in cfg.observables:
            # resolve limits and guess with fallback to defaults
            limits = cfg.limits.get(model, default_limits[model])
            guess = cfg.guess.get(model, default_guess[model])

            active_params = set(models_dict[model].params[obs]) - set(fixed.keys())
            init_guess = tuple(guess[p] for p in active_params)
            active_limits = {k: limits[k] for k in active_params}

            for redshift in ZS:
                for rv in RVS:
                    for vt in TYPES:
                        i += 1
                        print(f"\n [{i}/{Total}]")

                        # data_filename = f"{cfg.data['folder']}{cfg.data['prefix']}_Rv{rv}_z{redshift}_type{vt}_bin{cfg.binning}_map.fits"
                        chain_filename = Path(
                            f"{cfg.chain['folder']}fitting_{cfg.chain['sample']}_Rv{rv}_z{redshift}_type{vt}.hdf5"
                        ).expanduser()

                        print(" " + "-" * 25)
                        print(f" Model: {model}")
                        print(f" Profile: {obs}")
                        print(f" z: {redshift}")
                        print(f" Rv: {rv}")
                        print(f" Type: {vt}")
                        print(" " + "-" * 25)

                        sampler, L = run_emcee(
                            cosmo=cfg.cosmo,
                            NCORES=cfg.ncores,
                            NIT=cfg.nit,
                            NWALKERS=cfg.nwalkers,
                            burn_in=cfg.burn_in,
                            moves=cfg.moves,
                            data_filename=str(p[rv][redshift][vt]),
                            save_filename=chain_filename,
                            overwrite=cfg.overwrite,
                            model_name=model,
                            observable=obs,
                            cov_mode=cfg.cov_mode,
                            limits=active_limits,
                            fixed=fixed,
                            init_guess=init_guess,
                            pos_dist=cfg.pos_dist,
                            seed=cfg.seed,
                        )

                        discard = int(cfg.nit * cfg.discardp)
                        fitpar, errpar = get_fitted_params(
                            sampler.get_chain(discard=discard), active_params
                        )

                        chi2 = chi2_red(
                            L.ydata,
                            L.func(L.R, *fitpar.values()),
                            L.yerr,
                            len(L.R) - L.nparams,
                        )
                        print(" Fitted params:")
                        for (key, value), e in zip(fitpar.items(), errpar.values()):
                            print(
                                f"    > {key} = {value:.4g} ± ({e[0]:.4g},{e[1]:.4g})"
                            )
                        print(f"    > chi2_red = {chi2:4g}")

                        with h5py.File(chain_filename, "a") as f:
                            group_path = f"fitedparams/{model}/{obs}/{cfg.cov_mode}"

                            # Overwrite if exists
                            if (group_path in f) and (cfg.overwrite):
                                del f[group_path]

                            grp = f.create_group(group_path)

                            grp.attrs["chi2_red"] = chi2

                            for pname in active_params:
                                grp.attrs[f"{pname}_median"] = fitpar[pname]
                                grp.attrs[f"{pname}_err"] = np.array(errpar[pname])

                                # used limits to plot the correct posteriors
                                grp.attrs[f"{pname}_lim"] = np.array(
                                    active_limits[pname]
                                )

                        if cfg.do_plot:
                            plot_chains(sampler.get_chain(), labels=list(fitpar.keys()))
                            plot_corner(sampler, active_params, discard=discard)
                            plot_profile(
                                str(p[rv][redshift][vt]), obs, fitpar, model, color="k"
                            )

                            plt.show()


if __name__ == "__main__":
    from time import time

    print(" Start ".center(15, "#"))
    tini = time()
    main()
    print(" End :) ".center(15, "#"))
    print(f" >> Took {(time() - tini) / 60.0} min <<\n")
