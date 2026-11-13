import numpy as np
import toml
from pathlib import Path

from fitting.constants import get_cosmo


def parse_cosmoargs(cosmo_cfg):
    model = cosmo_cfg.get("model", "lcdm").lower()
    if model not in ["lcdm", "wcdm", "w0wacdm"]:
        raise ValueError(f"invalid model, got {model!r}.")

    params = cosmo_cfg[model]
    params.update({"model": model, "is_flat": cosmo_cfg["is_flat"]})

    return params


# for reading and autoclassify files
class FileClassifier:
    def __init__(
        self,
        prefix,
        gravity,
        threshold,
        label,
        nbins,
        rv_range,
        z_range,
        voidtype,
        binning,
        sourcemode,
    ):
        self.prefix = prefix
        self.gravity = gravity
        self.threshold = threshold
        self.label = label
        self.nbins = nbins
        self.rv_range = rv_range
        self.z_range = z_range
        self.voidtype = voidtype
        self.binning = binning
        self.sourcemode = sourcemode

        self.filename = "_".join(
            [
                prefix,
                gravity,
                threshold,
                label,
                nbins,
                rv_range,
                z_range,
                voidtype,
                binning,
                sourcemode,
            ]
        )

        self.clear_type()
        self.clear_sourcemode()

    def clear_type(self):
        self.voidtype = self.voidtype.removeprefix("type")

    def clear_sourcemode(self):
        self.sourcemode = self.sourcemode.removesuffix(".fits")


class Settings:
    def __init__(self, configfile):

        cfg = toml.load(configfile)

        self.data: dict = {
            "folder": cfg["data"]["folder"],
            "label": cfg["data"]["label"],
        }
        self.chain: dict = {
            "folder": cfg["chain"]["folder"],
            "prefix": "_".join(self.data["prefix"].split("_")[1:]),
            "sample": cfg["chain"]["sample"],
        }

        self.cosmo = get_cosmo(**parse_cosmoargs(cfg["cosmology"]))

        self.rv_ranges: list[str] = cfg["data"]["rv_ranges"]
        self.z_ranges: list[str] = cfg["data"]["z_ranges"]
        self.voidtypes: list[str] = cfg["data"]["voidtypes"]
        self.binning: str = cfg["data"]["binning"]

        self.ncores: int = cfg["run"]["ncores"]
        self.nit: int = cfg["run"]["nit"]
        self.burn_in: int = cfg["run"]["burn_in"]
        self.nwalkers: int = cfg["run"]["nwalkers"]
        self.do_plot: bool = cfg["run"]["do_plot"]
        self.overwrite: bool = cfg["run"]["overwrite"]

        self.cov_mode: str = cfg["fit"]["cov_mode"]
        self.observables: list = cfg["fit"]["observables"]
        self.models: list = cfg["fit"]["models"]
        self.pos_dist: str = cfg["fit"]["pos_dist"]
        self.seed: int = cfg["fit"]["seed"]
        self.discardp: float = cfg["fit"]["discardp"]
        self.moves: list[dict] = cfg["fit"]["moves"]

        raw_limits = cfg["fit"].get("limits", {})
        self.limits: dict = {
            model: {k: tuple(v) for k, v in params.items()}
            for model, params in raw_limits.items()
        }

        raw_guess = cfg["fit"].get("guess", {})
        self.guess: dict = {model: params for model, params in raw_guess.items()}
